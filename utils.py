from torch.utils.data import random_split
from gensim.models import KeyedVectors
import numpy as np
import torch 

def split_dataset(data_dict, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    total_size = len(data_dict)
    train_size = int(total_size * train_ratio)
    valid_size = int(total_size * valid_ratio)
    test_size = total_size - train_size - valid_size

    return random_split(data_dict, [train_size, valid_size, test_size])


def load_word2vec_model(model_path):
    word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    return word2vec_model

def create_embedding_matrix(word2vec_model, vocab, embedding_dim):
    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    for word, idx in vocab.items():
        if word in word2vec_model:
            embedding_vector = word2vec_model[word]
            embedding_matrix[idx] = embedding_vector
    return torch.tensor(embedding_matrix, dtype=torch.float)

class CustomError(Exception):
    def __init__(self, message):
        super().__init__(message)

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def check_word2vec_coverage(vocab, word2vec_model):
    covered = {}
    oov = {}
    covered_count = 0
    oov_count = 0

    for word in vocab.keys():
        if word in word2vec_model:
            covered[word] = word2vec_model[word]
            covered_count += 1
        else:
            oov[word] = vocab[word]
            oov_count += 1

    coverage = covered_count / (covered_count + oov_count)
    print("Number of words in vocab:", len(vocab))
    print("Number of words covered by Word2Vec:", covered_count)
    print("Number of words NOT covered by Word2Vec:", oov_count)
    print("Coverage of Word2Vec on vocab: {:.2f}%".format(coverage * 100))

    return covered, oov