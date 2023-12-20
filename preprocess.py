import pandas as pd
import jieba
import pinyin

import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
# from gensim.models import KeyedVectors

#from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.utils.rnn import pad_sequence

from utils import split_dataset, create_label_mapping, CustomError
from sklearn.feature_extraction.text import CountVectorizer

class Word2VecTextDataset(Dataset):
    def __init__(self, data_dict, vocab, label_mapping = None):
        self.data = data_dict
        self.vocab = vocab
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        text_indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in entry['processed_review'].split()]
        if self.label_mapping is not None:
            numerical_label = self.label_mapping[entry['cat']]
            labels = torch.tensor(numerical_label)
        else:
            labels = torch.tensor(entry['label'])

        return torch.tensor(text_indices), labels

def create_vocab(data_dict):
    all_tokens = [token for entry in data_dict for token in entry['processed_review'].split()]
    vocab = Counter(all_tokens)
    vocab_dict = {'<PAD>': 0, '<UNK>': 1}

    current_index = 2
    for word in vocab.keys():
        if word not in vocab_dict:
            vocab_dict[word] = current_index
            current_index += 1
    return vocab_dict

def lstm_collate_batch(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sequences, labels = zip(*batch)
    # lengths = [len(seq) for seq in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    label_list = torch.tensor(labels)
    # packed_sequences = pack_padded_sequence(padded_sequences, lengths, batch_first=True)
    return padded_sequences, label_list

def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, encoding="UTF-8").readlines()]
    return stopwords

def preprocess_text(text, stopwords, mode):

    if not isinstance(text, str):
        text = str(text)

    word_tokens = jieba.lcut(text)

    if stopwords is not None:
        word_tokens = [token for token in word_tokens if token.strip() and token not in stopwords]

    if mode == 'word':
        tokens = word_tokens
    elif mode == 'character':
        tokens = [char for token in word_tokens for char in token]
    elif mode == 'character_bigram':
        all_chars = ''.join(word_tokens) 
        tokens = [all_chars[i:i+2] for i in range(len(all_chars) - 1)]
    elif mode == 'pinyin':
        tokens = [pinyin.get(token, format="strip", delimiter=" ") for token in word_tokens]
        tokens = ' '.join(tokens).split()  # Flattening the list of pinyin strings
    else:
        raise ValueError("Wrong mode")
        
    if len(' '.join(tokens).split()) == 0:
        tokens = ['<UNK>']

    return ' '.join(tokens)

def pytorch_word2vec_dataloader(config):
    """
    This function creates and returns PyTorch dataloaders for word2vec training.
    
    Returns:
    train_dataloader (torch.utils.data.DataLoader): Dataloader for training data.
    valid_dataloader (torch.utils.data.DataLoader): Dataloader for validation data.
    test_dataloader (torch.utils.data.DataLoader): Dataloader for test data.
    vocab (list): List of unique words in the dataset.
    """
    dataframe = pd.read_csv(config.dataset_path)
    data_dict = dataframe.to_dict(orient='records')

    stopwords = None
    if config.use_stopwords:
        stopwords = stopwordslist(config.stopword_path)
    for entry in data_dict:
        entry['processed_review'] = preprocess_text(entry['review'], stopwords, config.mode)
    vocab = create_vocab(data_dict)
    
    label_mapping = None
    if config.n_classes > 2:
        label_mapping = create_label_mapping(dataframe)
    dataset = Word2VecTextDataset(data_dict, vocab, label_mapping)

    train_data, valid_data, test_data = split_dataset(dataset)

    train_dataloader = DataLoader(train_data, batch_size=config.batch_size, collate_fn=lstm_collate_batch)
    valid_dataloader = DataLoader(valid_data, batch_size=config.batch_size, collate_fn=lstm_collate_batch)
    test_dataloader = DataLoader(test_data, batch_size=config.batch_size, collate_fn=lstm_collate_batch)

    return train_dataloader, valid_dataloader, test_dataloader, vocab

class BagOfWordsTextDataset(Dataset):
    def __init__(self, data_dict, vectorizer, label_mapping = None):
        self.data = data_dict
        self.vectorizer = vectorizer
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        text = entry['processed_review']
        vector = self.vectorizer.transform([text]).toarray()
        if self.label_mapping is not None:
            numerical_label = self.label_mapping[entry['cat']]
            labels = torch.tensor(numerical_label)
        else:
            labels = torch.tensor(entry['label'])
        return torch.tensor(vector.flatten()), labels

def create_vectorizer(data_dict):
    texts = [entry['processed_review'] for entry in data_dict]
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    vectorizer.fit(texts)
    return vectorizer

def bag_of_words_collate_batch(batch):
    sequences, labels = zip(*batch)
    return torch.stack(sequences), torch.stack(labels)

def pytorch_bag_of_words_dataloader(config):
    dataframe = pd.read_csv(config.dataset_path)
    data_dict = dataframe.to_dict(orient='records')

    stopwords = None
    if config.use_stopwords:
        stopwords = stopwordslist(config.stopword_path)
    for entry in data_dict:
        entry['processed_review'] = preprocess_text(entry['review'], stopwords, config.mode)

    vectorizer = create_vectorizer(data_dict)

    label_mapping = None
    if config.n_classes > 2:
        label_mapping = create_label_mapping(dataframe)
    dataset = BagOfWordsTextDataset(data_dict, vectorizer, label_mapping)
    train_data, valid_data, test_data = split_dataset(dataset)

    train_dataloader = DataLoader(train_data, batch_size=config.batch_size, collate_fn=bag_of_words_collate_batch)
    valid_dataloader = DataLoader(valid_data, batch_size=config.batch_size, collate_fn=bag_of_words_collate_batch)
    test_dataloader = DataLoader(test_data, batch_size=config.batch_size, collate_fn=bag_of_words_collate_batch)

    return train_dataloader, valid_dataloader, test_dataloader, vectorizer.get_feature_names_out()

if __name__ == "__main__":
    dataloader = pytorch_word2vec_dataloader()
    it = iter(dataloader)
    first = next(it)
    second = next(it)
    print(first)
    print(second)