import pandas as pd
import jieba
import pinyin

from config import Config

import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from gensim.models import KeyedVectors

#from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from utils import *

class Word2VecTextDataset(Dataset):
    def __init__(self, data_dict, vocab):
        self.data = data_dict
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        text_indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in entry['processed_review'].split()]
        return torch.tensor(text_indices), torch.tensor(entry['label'])

def create_vocab(data_dict):
    all_tokens = [token for entry in data_dict for token in entry['processed_review'].split()]
    vocab = Counter(all_tokens)
    vocab = {word: idx for idx, (word, _) in enumerate(vocab.items(), start=2)} 
    vocab['<PAD>'] = 0  
    vocab['<UNK>'] = 1 
    return vocab

def lstm_collate_batch(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    label_list = torch.tensor(labels)
    packed_sequences = pack_padded_sequence(padded_sequences, lengths, batch_first=True)

    return packed_sequences, label_list

def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, encoding="UTF-8").readlines()]
    return stopwords

def preprocess_text(text, stopwords, mode=Config.mode):

    if not isinstance(text, str):
        text = str(text)

    tokens = []

    if mode == 'word':
        tokens = jieba.lcut(text)
    elif mode == 'character':
        tokens = [char for char in text]
    elif mode == 'pingyin':
        tokens = pinyin.get(text, format="strip", delimiter=" ").split()
    else:
        raise CustomError("Wrong mode")


    tokens = [token for token in tokens if token not in stopwords]

    return ' '.join(tokens)

def pytorch_word2vec_dataloader():
    dataframe = pd.read_csv(Config.dataset_path)
    data_dict = dataframe.to_dict(orient='records')

    stopwords = stopwordslist(Config.stopword_path)
    for entry in data_dict:
        entry['processed_review'] = preprocess_text(entry['review'], stopwords)

    vocab = create_vocab(data_dict)
    dataset = Word2VecTextDataset(data_dict, vocab)
    train_data, valid_data, test_data = split_dataset(dataset)

    train_dataloader = DataLoader(train_data, batch_size=Config.batch_size, collate_fn=lstm_collate_batch)
    valid_dataloader = DataLoader(valid_data, batch_size=Config.batch_size, collate_fn=lstm_collate_batch)
    test_dataloader = DataLoader(test_data, batch_size=Config.batch_size, collate_fn=lstm_collate_batch)

    return train_dataloader, valid_dataloader, test_dataloader, vocab

###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
"""
class BERTTextDataset(Dataset):
    def __init__(self, data_dict, bert_model_name='bert-base-uncased'):
        self.data = data_dict
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        encoding = self.tokenizer.encode_plus(
            entry['processed_review'],
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(entry['label'])
        }

def bert_collate_batch(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_masks = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return input_ids, attention_masks, labels

def pytorch_bert_dataloader():
    dataframe = pd.read_csv(Config.dataset_path)
    data_dict = dataframe.to_dict(orient='records')

    stopwords = stopwordslist(Config.stopword_path)
    for entry in data_dict:
        entry['processed_review'] = preprocess_text(entry['review'], stopwords)
        
    dataset = BERTTextDataset(data_dict)
    train_data, valid_data, test_data = split_dataset(dataset)

    train_dataloader = DataLoader(train_data, batch_size=Config.batch_size, collate_fn=bert_collate_batch)
    valid_dataloader = DataLoader(valid_data, batch_size=Config.batch_size, collate_fn=lstm_collate_batch)
    test_dataloader = DataLoader(test_data, batch_size=Config.batch_size, collate_fn=lstm_collate_batch)

    return train_dataloader, valid_dataloader, test_dataloader

"""
if __name__ == "__main__":
    dataloader = pytorch_word2vec_dataloader()
    it = iter(dataloader)
    first = next(it)
    second = next(it)
    print(first)
    print(second)

