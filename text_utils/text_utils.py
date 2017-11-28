import numpy as np
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from scipy.sparse import dok_matrix

from collections import Counter, defaultdict, OrderedDict, namedtuple
import copy

en_stopwords = set(stopwords.words('english'))

def not_stopword(word):
    return word.lower() not in en_stopwords

def not_all_punct(word):
    return not all([not c.isalnum() for c in word])

def to_lower(word):
    return word.lower()

snowball = SnowballStemmer('english')
def stemmer(word):
    return snowball.stem(word)

class Tokenizer(object):
    def __init__(self,
            transforms=None,
            filters=None,
            word_tokenizer=None,
            sent_tokenizer=None):

        if transforms is None:
            transforms = []
        if word_tokenizer is None:
            word_tokenizer = word_tokenize
        if filters is None:
            filters = []
        self.word_tokenizer = word_tokenizer
        self.sent_tokenizer = sent_tokenizer
        self.filters = filters
        self.transforms = transforms

    def tokenize(self, str_to_tokenize):
        if self.sent_tokenizer is None:
            sentences = [str_to_tokenize]
        else:
            sentences = self.sent_tokenizer(str_to_tokenize)

        tokens_by_sentence = []
        for sent in sentences:
            tokens = []
            for t in self.word_tokenizer(sent):
                include_token = all(f(t) for f in self.filters)
                if include_token:
                    transformed = t
                    for tr in self.transforms:
                        transformed = tr(t)
                    tokens.append(transformed)
                    
            tokens_by_sentence.append(tokens)

        if self.sent_tokenizer is None:
            return tokens_by_sentence[0]
        else:
            return tokens_by_sentence

default_tokenizer = Tokenizer(
    filters = [not_all_punct],
    transforms = [to_lower],
    word_tokenizer = word_tokenize,
    sent_tokenizer = None)

def xml_to_sparse_term_doc(filename, within, eachdoc, parser_type, tokenizer):
    docs = xml_to_document_strings(filename, within, eachdoc, parser_type)
    return document_strings_to_sparse_term_doc(docs, tokenizer)

def xml_to_document_strings(filename, within, eachdoc, parser_type):
    within = copy.copy(within)
    with open(filename, 'r') as f:
        soup = BeautifulSoup(f, features=parser_type)
    within_tag = soup
    while len(within) > 0:
        within_tag = within_tag.find(within[0], recursive=False)
        within = within[1:]

    doc_strings = []
    doc_top_levels = within_tag.find_all(eachdoc[0], recursive=False)
    for doc in doc_top_levels:
        current_tag = doc
        current_remaining = copy.copy(eachdoc[1:])
        while len(current_remaining) > 0:
            current_tag = current_tag.find(current_remaining[0], recursive=False)
            current_remaining = current_remaining[1:]
        doc_strings.append(current_tag.get_text(strip=True))

    return doc_strings

def document_strings_to_sparse_term_doc(docs, tokenizer, vocab = None):
    # Tokenize and build up vocabulary
    if vocab is None:
        vocab = []
        all_token_indices = dict()
    else:
        all_token_indices = {v: i for i, v in enumerate(vocab)}
    token_indices_by_document = []
    for doc in docs:
        tokens = tokenizer.tokenize(doc)
        current_token_indices = []
        for t in tokens:
            if t not in all_token_indices:
                all_token_indices[t] = len(all_token_indices)
                vocab.append(t)
            current_token_indices.append(all_token_indices[t])
        token_indices_by_document.append(current_token_indices)

    # Construct sparse term-document matrix
    vocab_size = len(vocab)
    num_docs = len(token_indices_by_document)
    term_doc_matrix = dok_matrix((vocab_size, num_docs), dtype='int')
    for doc_index in range(num_docs):
        for term_index in token_indices_by_document[doc_index]:
            term_doc_matrix[term_index, doc_index] += 1
    term_doc_matrix = term_doc_matrix.tocsc()

    return term_doc_matrix, vocab
