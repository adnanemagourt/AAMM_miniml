import numpy as np
import re
from scipy.sparse import csr_matrix
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from unicodedata import normalize


import sys
sys.path.append('..')
from _BaseClasses import Transformer


class CountVectorizer(Transformer):
    """
    Convert a collection of text documents to a matrix of token counts.
    
    Parameters
    ----------
    
    input : {'content'}, default='content'
        If 'content', the input will be treated as a list of items, where each item is a string.
    encoding : str, default='utf-8'
        The encoding of the text.
    decode_error : {'strict', 'ignore', 'replace'}, default='strict'
        Instruction on what to do if a byte sequence is invalid.
    strip_accents : {'ascii', 'unicode'}, default=None
        Remove accents and perform other character normalization during the preprocessing step.
    lowercase : bool, default=True
        Convert all characters to lowercase before tokenizing.
    preprocessor : callable, default=None
        Override the preprocessing (string transformation) stage while preserving the tokenizing and n-grams generation steps.
    tokenizer : callable, default=None
        Override the string tokenization step while preserving the preprocessing and n-grams generation steps.
    stop_words : {'english'}, list, default='english'
        Remove common words if 'english', or use a custom list of stop words.
    token_pattern : str, default=r"(?u)\b\w\w+\b
        Regular expression denoting what constitutes a "token".
    ngram_range : tuple, default=(1, 1)
        The lower and upper boundary of the range of n-values for different n-grams to be extracted.    
    analyzer : {'word'}, default='word'
        Whether the feature should be made of word n-gram or character n-grams.
    max_df : float, default=1.0 
        When building the vocabulary, ignore terms that have a document frequency strictly higher than the given threshold.
    min_df : int, default=1 
        When building the vocabulary, ignore terms that have a document frequency strictly lower than the given threshold.
    max_features : int, default=None
        If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.
    vocabulary : dict, default=None
        Mapping of terms to feature indices.
    binary : bool, default=False
        If True, all non-zero counts are set to 1.
    dtype : type, default=np.int64
        Type of the matrix returned by fit_transform() or transform().
    
    """
    def __init__(self, *, input='content', encoding='utf-8', decode_error='strict',
                 strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words="english", token_pattern=r"(?u)\b\w\w+\b", ngram_range=(1, 1),
                 analyzer='word', max_df=1.0, min_df=1, max_features=None, vocabulary=None,
                 binary=False, dtype=np.int64):
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.lowercase = lowercase
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.token_pattern = token_pattern
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.vocabulary = vocabulary
        self.binary = binary
        self.dtype = dtype
        self.fixed_vocabulary_ = False
        self.vocabulary_ = {}
        self.stop_words_ = set()

    def _build_analyzer(self):
        preprocessor = self.build_preprocessor()
        tokenizer = self.build_tokenizer()
        stop_words = self.get_stop_words()
        
        def analyzer(doc):
            doc = preprocessor(doc)
            tokens = tokenizer(doc)
            if self.ngram_range == (1, 1):
                return [token for token in tokens if token not in stop_words]
            else:
                return [ngram for ngram in self._word_ngrams(tokens) if ngram not in stop_words]
        
        return analyzer

    def build_preprocessor(self):
        if self.preprocessor is not None:
            return self.preprocessor
        
        def preprocess(doc):
            if self.lowercase:
                doc = doc.lower()
            if self.strip_accents:
                doc = self.strip_accents_method(doc)
            return doc
        
        return preprocess

    def build_tokenizer(self):
        if self.tokenizer is not None:
            return self.tokenizer
        
        token_pattern = re.compile(self.token_pattern)
        
        def tokenize(doc):
            return token_pattern.findall(doc)
        
        return tokenize

    def strip_accents_method(self, doc):
        if self.strip_accents == 'ascii':
            return normalize('NFKD', doc).encode('ascii', 'ignore').decode('ascii')
        elif self.strip_accents == 'unicode':
            return normalize('NFKD', doc)
        return doc

    def fit(self, raw_documents, y=None):
        self._validate_params()
        self.fixed_vocabulary_ = self.vocabulary is not None
        
        if self.fixed_vocabulary_:
            self.vocabulary_ = self.vocabulary
        else:
            vocabulary = {}
            analyzer = self._build_analyzer()
            
            for doc in raw_documents:
                for feature in analyzer(doc):
                    if feature not in vocabulary:
                        vocabulary[feature] = len(vocabulary)
            
            # Sort vocabulary by token order as they appear in the documents
            self.vocabulary_ = dict(sorted(vocabulary.items(), key=lambda item: item[1]))
        
        return self

    def fit_transform(self, raw_documents, y=None):
        self.fit(raw_documents, y)
        return self.transform(raw_documents)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'vocabulary_')
        if input_features is not None:
            raise ValueError("input_features is not supported")
        return np.array(sorted(self.vocabulary_, key=lambda x: self.vocabulary_[x]))

    def get_stop_words(self):
        if self.stop_words == 'english':
            return ENGLISH_STOP_WORDS
        return self.stop_words

    def transform(self, raw_documents):
        check_is_fitted(self, 'vocabulary_')
        analyzer = self._build_analyzer()
        rows, cols, data = [], [], []
        
        for i, doc in enumerate(raw_documents):
            feature_counts = {}
            for feature in analyzer(doc):
                if feature in self.vocabulary_:
                    idx = self.vocabulary_[feature]
                    if idx not in feature_counts:
                        feature_counts[idx] = 0
                    feature_counts[idx] += 1
            
            for idx, count in feature_counts.items():
                rows.append(i)
                cols.append(idx)
                data.append(1 if self.binary else count)
        
        shape = (len(raw_documents), len(self.vocabulary_))
        return csr_matrix((data, (rows, cols)), shape=shape, dtype=self.dtype)

    def _word_ngrams(self, tokens):
        min_n, max_n = self.ngram_range
        if max_n == 1:
            return tokens
        ngrams = []
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngrams.append(' '.join(tokens[i: i + n]))
        return ngrams

    def _validate_params(self):
        if isinstance(self.ngram_range, tuple) and len(self.ngram_range) == 2:
            if not all(isinstance(n, int) for n in self.ngram_range):
                raise ValueError("ngram_range should be a tuple of two integers")
        else:
            raise ValueError("ngram_range should be a tuple of two integers")






# class CountVectorizer(BaseEstimator, TransformerMixin):
#     def __init__(self, *, input='content', encoding='utf-8', decode_error='strict',
#                  strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None,
#                  stop_words="english", token_pattern=r"(?u)\b\w\w+\b", ngram_range=(1, 1),
#                  analyzer='word', max_df=1.0, min_df=1, max_features=None, vocabulary=None,
#                  binary=False, dtype=np.int64):
#         self.input = input
#         self.encoding = encoding
#         self.decode_error = decode_error
#         self.strip_accents = strip_accents
#         self.lowercase = lowercase
#         self.preprocessor = preprocessor
#         self.tokenizer = tokenizer
#         self.stop_words = stop_words
#         self.token_pattern = token_pattern
#         self.ngram_range = ngram_range
#         self.analyzer = analyzer
#         self.max_df = max_df
#         self.min_df = min_df
#         self.max_features = max_features
#         self.vocabulary = vocabulary
#         self.binary = binary
#         self.dtype = dtype
#         self.fixed_vocabulary_ = False
#         self.vocabulary_ = {}
#         self.stop_words_ = set()

#     def build_analyzer(self):
#         preprocessor = self.build_preprocessor()
#         tokenizer = self.build_tokenizer()
#         stop_words = self.get_stop_words()
        
#         def analyzer(doc):
#             doc = preprocessor(doc)
#             tokens = tokenizer(doc)
#             tokens = [token for token in tokens if token not in stop_words]
#             if self.ngram_range == (1, 1):
#                 return tokens
#             else:
#                 return [ngram for ngram in self._word_ngrams(tokens)]
        
#         return analyzer

#     def build_preprocessor(self):
#         if self.preprocessor is not None:
#             return self.preprocessor
        
#         def preprocess(doc):
#             if self.lowercase:
#                 doc = doc.lower()
#             if self.strip_accents:
#                 doc = self.strip_accents_method(doc)
#             return doc
        
#         return preprocess

#     def build_tokenizer(self):
#         if self.tokenizer is not None:
#             return self.tokenizer
        
#         token_pattern = re.compile(self.token_pattern)
        
#         def tokenize(doc):
#             return token_pattern.findall(doc)
        
#         return tokenize

#     def strip_accents_method(self, doc):
#         if self.strip_accents == 'ascii':
#             return normalize('NFKD', doc).encode('ascii', 'ignore').decode('ascii')
#         elif self.strip_accents == 'unicode':
#             return normalize('NFKD', doc)
#         return doc

#     def decode(self, doc):
#         if isinstance(doc, bytes):
#             return doc.decode(self.encoding, self.decode_error)
#         return doc

#     def fit(self, raw_documents, y=None):
#         self._validate_params()
#         self.fixed_vocabulary_ = self.vocabulary is not None
        
#         if self.fixed_vocabulary_:
#             self.vocabulary_ = self.vocabulary
#         else:
#             vocabulary = {}
#             analyzer = self.build_analyzer()
            
#             for doc in raw_documents:
#                 for feature in analyzer(doc):
#                     if feature not in vocabulary:
#                         vocabulary[feature] = len(vocabulary)
            
#             self.vocabulary_ = vocabulary
        
#         return self

#     def fit_transform(self, raw_documents, y=None):
#         self.fit(raw_documents, y)
#         return self.transform(raw_documents)

#     def get_feature_names_out(self, input_features=None):
#         check_is_fitted(self, 'vocabulary_')
#         if input_features is not None:
#             raise ValueError("input_features is not supported")
#         return np.array(sorted(self.vocabulary_, key=lambda x: self.vocabulary_[x]))

#     def get_stop_words(self):
#         if self.stop_words == 'english':
#             return ENGLISH_STOP_WORDS
#         return self.stop_words

#     def inverse_transform(self, X):
#         check_is_fitted(self, 'vocabulary_')
#         return [[self.vocabulary_[i] for i in np.where(doc)[0]] for doc in X]

#     def set_fit_request(self, *, raw_documents: bool | None | str = '$UNCHANGED$'):
#         self._fit_request = raw_documents
#         return self

#     def set_params(self, **params):
#         for key, value in params.items():
#             setattr(self, key, value)
#         return self

#     def set_transform_request(self, *, raw_documents: bool | None | str = '$UNCHANGED$'):
#         self._transform_request = raw_documents
#         return self

#     def transform(self, raw_documents):
#         check_is_fitted(self, 'vocabulary_')
#         analyzer = self.build_analyzer()
#         rows, cols, data = [], [], []
        
#         for i, doc in enumerate(raw_documents):
#             feature_counts = {}
#             for feature in analyzer(doc):
#                 if feature in self.vocabulary_:
#                     feature_index = self.vocabulary_[feature]
#                     if feature_index not in feature_counts:
#                         feature_counts[feature_index] = 0
#                     feature_counts[feature_index] += 1
            
#             for feature_index, count in feature_counts.items():
#                 rows.append(i)
#                 cols.append(feature_index)
#                 data.append(1 if self.binary else count)
        
#         shape = (len(raw_documents), len(self.vocabulary_))
#         return csr_matrix((data, (rows, cols)), shape=shape, dtype=self.dtype)

#     def _word_ngrams(self, tokens):
#         min_n, max_n = self.ngram_range
#         if max_n == 1:
#             return tokens
#         ngrams = []
#         for n in range(min_n, max_n + 1):
#             for i in range(len(tokens) - n + 1):
#                 ngrams.append(' '.join(tokens[i: i + n]))
#         return ngrams

#     def _validate_params(self):
#         if isinstance(self.ngram_range, tuple) and len(self.ngram_range) == 2:
#             if not all(isinstance(n, int) for n in self.ngram_range):
#                 raise ValueError("ngram_range should be a tuple of two integers")
#         else:
#             raise ValueError("ngram_range should be a tuple of two integers")





# class CountVectorizer(BaseEstimator, TransformerMixin):
#     def __init__(self, *, input='content', encoding='utf-8', decode_error='strict',
#                  strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None,
#                  stop_words="english", token_pattern=r"(?u)\b\w\w+\b", ngram_range=(1, 1),
#                  analyzer='word', max_df=1.0, min_df=1, max_features=None, vocabulary=None,
#                  binary=False, dtype=np.int64):
#         self.input = input
#         self.encoding = encoding
#         self.decode_error = decode_error
#         self.strip_accents = strip_accents
#         self.lowercase = lowercase
#         self.preprocessor = preprocessor
#         self.tokenizer = tokenizer
#         self.stop_words = stop_words
#         self.token_pattern = token_pattern
#         self.ngram_range = ngram_range
#         self.analyzer = analyzer
#         self.max_df = max_df
#         self.min_df = min_df
#         self.max_features = max_features
#         self.vocabulary = vocabulary
#         self.binary = binary
#         self.dtype = dtype
#         self.fixed_vocabulary_ = False
#         self.vocabulary_ = {}
#         self.stop_words_ = set()

#     def build_analyzer(self):
#         preprocessor = self.build_preprocessor()
#         tokenizer = self.build_tokenizer()
#         stop_words = self.get_stop_words()
        
#         def analyzer(doc):
#             doc = preprocessor(doc)
#             tokens = tokenizer(doc)
#             if self.ngram_range == (1, 1):
#                 return [token for token in tokens if token not in stop_words]
#             else:
#                 return [ngram for ngram in self._word_ngrams(tokens) if ngram not in stop_words]
        
#         return analyzer

#     def build_preprocessor(self):
#         if self.preprocessor is not None:
#             return self.preprocessor
        
#         def preprocess(doc):
#             if self.lowercase:
#                 doc = doc.lower()
#             if self.strip_accents:
#                 doc = self.strip_accents_method(doc)
#             return doc
        
#         return preprocess

#     def build_tokenizer(self):
#         if self.tokenizer is not None:
#             return self.tokenizer
        
#         token_pattern = re.compile(self.token_pattern)
        
#         def tokenize(doc):
#             return token_pattern.findall(doc)
        
#         return tokenize

#     def strip_accents_method(self, doc):
#         if self.strip_accents == 'ascii':
#             return normalize('NFKD', doc).encode('ascii', 'ignore').decode('ascii')
#         elif self.strip_accents == 'unicode':
#             return normalize('NFKD', doc)
#         return doc

#     def decode(self, doc):
#         if isinstance(doc, bytes):
#             return doc.decode(self.encoding, self.decode_error)
#         return doc

#     def fit(self, raw_documents, y=None):
#         self._validate_params()
#         self.fixed_vocabulary_ = self.vocabulary is not None
        
#         if self.fixed_vocabulary_:
#             self.vocabulary_ = self.vocabulary
#         else:
#             vocabulary = {}
#             analyzer = self.build_analyzer()
            
#             for doc in raw_documents:
#                 for feature in analyzer(doc):
#                     if feature not in vocabulary:
#                         vocabulary[feature] = len(vocabulary)
            
#             self.vocabulary_ = vocabulary
        
#         return self

#     def fit_transform(self, raw_documents, y=None):
#         self.fit(raw_documents, y)
#         return self.transform(raw_documents)

#     def get_feature_names_out(self, input_features=None):
#         check_is_fitted(self, 'vocabulary_')
#         if input_features is not None:
#             raise ValueError("input_features is not supported")
#         return np.array(sorted(self.vocabulary_, key=lambda x: self.vocabulary_[x]))

#     def get_metadata_routing(self):
#         return MetadataRequest(self)

#     def get_params(self, deep=True):
#         return self.__dict__

#     def get_stop_words(self):
#         if self.stop_words == 'english':
#             return ENGLISH_STOP_WORDS
#         return self.stop_words

#     def inverse_transform(self, X):
#         check_is_fitted(self, 'vocabulary_')
#         return [[self.vocabulary_[i] for i in np.where(doc)[0]] for doc in X]

#     def set_fit_request(self, *, raw_documents: bool | None | str = '$UNCHANGED$'):
#         self._fit_request = raw_documents
#         return self

#     def set_params(self, **params):
#         for key, value in params.items():
#             setattr(self, key, value)
#         return self

#     def set_transform_request(self, *, raw_documents: bool | None | str = '$UNCHANGED$'):
#         self._transform_request = raw_documents
#         return self

#     def transform(self, raw_documents):
#         check_is_fitted(self, 'vocabulary_')
#         analyzer = self.build_analyzer()
#         rows, cols, data = [], [], []
        
#         for i, doc in enumerate(raw_documents):
#             feature_counts = {}
#             for feature in analyzer(doc):
#                 if feature in self.vocabulary_:
#                     feature_index = self.vocabulary_[feature]
#                     if feature_index not in feature_counts:
#                         feature_counts[feature_index] = 0
#                     feature_counts[feature_index] += 1
            
#             for feature_index, count in feature_counts.items():
#                 rows.append(i)
#                 cols.append(feature_index)
#                 data.append(1 if self.binary else count)
        
#         shape = (len(raw_documents), len(self.vocabulary_))
#         return csr_matrix((data, (rows, cols)), shape=shape, dtype=self.dtype)

#     def _word_ngrams(self, tokens):
#         min_n, max_n = self.ngram_range
#         if max_n == 1:
#             return tokens
#         ngrams = []
#         for n in range(min_n, max_n + 1):
#             for i in range(len(tokens) - n + 1):
#                 ngrams.append(' '.join(tokens[i: i + n]))
#         return ngrams

#     def _validate_params(self):
#         if isinstance(self.ngram_range, tuple) and len(self.ngram_range) == 2:
#             if not all(isinstance(n, int) for n in self.ngram_range):
#                 raise ValueError("ngram_range should be a tuple of two integers")
#         else:
#             raise ValueError("ngram_range should be a tuple of two integers")





# class CountVectorizer(BaseEstimator, TransformerMixin):
#     def __init__(self, *, input='content', encoding='utf-8', decode_error='strict',
#                  strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None,
#                  stop_words="english", token_pattern=r"(?u)\b\w\w+\b", ngram_range=(1, 1),
#                  analyzer='word', max_df=1.0, min_df=1, max_features=None, vocabulary=None,
#                  binary=False, dtype=np.int64):
#         self.input = input
#         self.encoding = encoding
#         self.decode_error = decode_error
#         self.strip_accents = strip_accents
#         self.lowercase = lowercase
#         self.preprocessor = preprocessor
#         self.tokenizer = tokenizer
#         self.stop_words = stop_words
#         self.token_pattern = token_pattern
#         self.ngram_range = ngram_range
#         self.analyzer = analyzer
#         self.max_df = max_df
#         self.min_df = min_df
#         self.max_features = max_features
#         self.vocabulary = vocabulary
#         self.binary = binary
#         self.dtype = dtype
#         self.fixed_vocabulary_ = False
#         self.vocabulary_ = {}
#         self.stop_words_ = set()

#     def _build_analyzer(self):
#         preprocessor = self.build_preprocessor()
#         tokenizer = self.build_tokenizer()
#         stop_words = self.get_stop_words()
        
#         def analyzer(doc):
#             doc = preprocessor(doc)
#             tokens = tokenizer(doc)
#             if self.ngram_range == (1, 1):
#                 return [token for token in tokens if token not in stop_words]
#             else:
#                 return [ngram for ngram in self._word_ngrams(tokens) if ngram not in stop_words]
        
#         return analyzer

#     def build_preprocessor(self):
#         if self.preprocessor is not None:
#             return self.preprocessor
        
#         def preprocess(doc):
#             if self.lowercase:
#                 doc = doc.lower()
#             if self.strip_accents:
#                 doc = self.strip_accents_method(doc)
#             return doc
        
#         return preprocess

#     def build_tokenizer(self):
#         if self.tokenizer is not None:
#             return self.tokenizer
        
#         token_pattern = re.compile(self.token_pattern)
        
#         def tokenize(doc):
#             return token_pattern.findall(doc)
        
#         return tokenize

#     def strip_accents_method(self, doc):
#         if self.strip_accents == 'ascii':
#             return normalize('NFKD', doc).encode('ascii', 'ignore').decode('ascii')
#         elif self.strip_accents == 'unicode':
#             return normalize('NFKD', doc)
#         return doc

#     def fit(self, raw_documents, y=None):
#         self._validate_params()
#         self.fixed_vocabulary_ = self.vocabulary is not None
        
#         if self.fixed_vocabulary_:
#             self.vocabulary_ = self.vocabulary
#         else:
#             vocabulary = {}
#             analyzer = self._build_analyzer()
            
#             for doc in raw_documents:
#                 for feature in analyzer(doc):
#                     if feature not in vocabulary:
#                         vocabulary[feature] = len(vocabulary)
            
#             self.vocabulary_ = vocabulary
        
#         return self

#     def fit_transform(self, raw_documents, y=None):
#         self.fit(raw_documents, y)
#         return self.transform(raw_documents)

#     def get_feature_names_out(self, input_features=None):
#         check_is_fitted(self, 'vocabulary_')
#         if input_features is not None:
#             raise ValueError("input_features is not supported")
#         return np.array(sorted(self.vocabulary_, key=lambda x: self.vocabulary_[x]))

#     def get_stop_words(self):
#         if self.stop_words == 'english':
#             return ENGLISH_STOP_WORDS
#         return self.stop_words

#     def transform(self, raw_documents):
#         check_is_fitted(self, 'vocabulary_')
#         analyzer = self._build_analyzer()
#         rows, cols, data = [], [], []
        
#         for i, doc in enumerate(raw_documents):
#             feature_counts = {}
#             for feature in analyzer(doc):
#                 if feature in self.vocabulary_:
#                     idx = self.vocabulary_[feature]
#                     if idx not in feature_counts:
#                         feature_counts[idx] = 0
#                     feature_counts[idx] += 1
            
#             for idx, count in feature_counts.items():
#                 rows.append(i)
#                 cols.append(idx)
#                 data.append(1 if self.binary else count)
        
#         shape = (len(raw_documents), len(self.vocabulary_))
#         return csr_matrix((data, (rows, cols)), shape=shape, dtype=self.dtype)

#     def _word_ngrams(self, tokens):
#         min_n, max_n = self.ngram_range
#         if max_n == 1:
#             return tokens
#         ngrams = []
#         for n in range(min_n, max_n + 1):
#             for i in range(len(tokens) - n + 1):
#                 ngrams.append(' '.join(tokens[i: i + n]))
#         return ngrams

#     def _validate_params(self):
#         if isinstance(self.ngram_range, tuple) and len(self.ngram_range) == 2:
#             if not all(isinstance(n, int) for n in self.ngram_range):
#                 raise ValueError("ngram_range should be a tuple of two integers")
#         else:
#             raise ValueError("ngram_range should be a tuple of two integers")





# class CountVectorizer(BaseEstimator, TransformerMixin):
#     def __init__(self, *, input='content', encoding='utf-8', decode_error='strict',
#                  strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None,
#                  stop_words="english", token_pattern=r"(?u)\b\w\w+\b", ngram_range=(1, 1),
#                  analyzer='word', max_df=1.0, min_df=1, max_features=None, vocabulary=None,
#                  binary=False, dtype=np.int64):
#         self.input = input
#         self.encoding = encoding
#         self.decode_error = decode_error
#         self.strip_accents = strip_accents
#         self.lowercase = lowercase
#         self.preprocessor = preprocessor
#         self.tokenizer = tokenizer
#         self.stop_words = stop_words
#         self.token_pattern = token_pattern
#         self.ngram_range = ngram_range
#         self.analyzer = analyzer
#         self.max_df = max_df
#         self.min_df = min_df
#         self.max_features = max_features
#         self.vocabulary = vocabulary
#         self.binary = binary
#         self.dtype = dtype
#         self.fixed_vocabulary_ = False
#         self.vocabulary_ = {}
#         self.stop_words_ = set()

#     def build_analyzer(self):
#         preprocessor = self.build_preprocessor()
#         tokenizer = self.build_tokenizer()
#         stop_words = self.get_stop_words()
        
#         def analyzer(doc):
#             doc = preprocessor(doc)
#             tokens = tokenizer(doc)
#             if self.ngram_range == (1, 1):
#                 return [token for token in tokens if token not in stop_words]
#             else:
#                 return [ngram for ngram in self._word_ngrams(tokens) if ngram not in stop_words]
        
#         return analyzer

#     def build_preprocessor(self):
#         if self.preprocessor is not None:
#             return self.preprocessor
        
#         def preprocess(doc):
#             if self.lowercase:
#                 doc = doc.lower()
#             if self.strip_accents:
#                 doc = self.strip_accents_method(doc)
#             return doc
        
#         return preprocess

#     def build_tokenizer(self):
#         if self.tokenizer is not None:
#             return self.tokenizer
        
#         token_pattern = re.compile(self.token_pattern)
        
#         def tokenize(doc):
#             return token_pattern.findall(doc)
        
#         return tokenize

#     def strip_accents_method(self, doc):
#         if self.strip_accents == 'ascii':
#             return normalize('NFKD', doc).encode('ascii', 'ignore').decode('ascii')
#         elif self.strip_accents == 'unicode':
#             return normalize('NFKD', doc)
#         return doc

#     def decode(self, doc):
#         if isinstance(doc, bytes):
#             return doc.decode(self.encoding, self.decode_error)
#         return doc

#     def fit(self, raw_documents, y=None):
#         self._validate_params()
#         self.fixed_vocabulary_ = self.vocabulary is not None
        
#         if self.fixed_vocabulary_:
#             self.vocabulary_ = self.vocabulary
#         else:
#             vocabulary = {}
#             analyzer = self.build_analyzer()
            
#             for doc in raw_documents:
#                 for feature in analyzer(doc):
#                     if feature not in vocabulary:
#                         vocabulary[feature] = len(vocabulary)
            
#             self.vocabulary_ = vocabulary
        
#         return self

#     def fit_transform(self, raw_documents, y=None):
#         self.fit(raw_documents, y)
#         return self.transform(raw_documents)

#     def get_feature_names_out(self, input_features=None):
#         check_is_fitted(self, 'vocabulary_')
#         if input_features is not None:
#             raise ValueError("input_features is not supported")
#         return np.array(sorted(self.vocabulary_, key=lambda x: self.vocabulary_[x]))

#     def get_metadata_routing(self):
#         return MetadataRequest(self)

#     def get_params(self, deep=True):
#         return self.__dict__

#     def get_stop_words(self):
#         if self.stop_words == 'english':
#             return ENGLISH_STOP_WORDS
#         return self.stop_words

#     def inverse_transform(self, X):
#         check_is_fitted(self, 'vocabulary_')
#         return [[self.vocabulary_[i] for i in np.where(doc)[0]] for doc in X]

#     def set_fit_request(self, *, raw_documents: bool | None | str = '$UNCHANGED$'):
#         self._fit_request = raw_documents
#         return self

#     def set_params(self, **params):
#         for key, value in params.items():
#             setattr(self, key, value)
#         return self

#     def set_transform_request(self, *, raw_documents: bool | None | str = '$UNCHANGED$'):
#         self._transform_request = raw_documents
#         return self

#     def transform(self, raw_documents):
#         check_is_fitted(self, 'vocabulary_')
#         analyzer = self.build_analyzer()
#         rows, cols, data = [], [], []
        
#         for i, doc in enumerate(raw_documents):
#             for feature in analyzer(doc):
#                 if feature in self.vocabulary_:
#                     rows.append(i)
#                     cols.append(self.vocabulary_[feature])
#                     data.append(1 if self.binary else data.count(feature))
        
#         shape = (len(raw_documents), len(self.vocabulary_))
#         return csr_matrix((data, (rows, cols)), shape=shape, dtype=self.dtype)

#     def _word_ngrams(self, tokens):
#         min_n, max_n = self.ngram_range
#         if max_n == 1:
#             return tokens
#         ngrams = []
#         for n in range(min_n, max_n + 1):
#             for i in range(len(tokens) - n + 1):
#                 ngrams.append(' '.join(tokens[i: i + n]))
#         return ngrams

#     def _validate_params(self):
#         if isinstance(self.ngram_range, tuple) and len(self.ngram_range) == 2:
#             if not all(isinstance(n, int) for n in self.ngram_range):
#                 raise ValueError("ngram_range should be a tuple of two integers")
#         else:
#             raise ValueError("ngram_range should be a tuple of two integers")
