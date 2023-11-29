import jieba

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch
from torch.utils.data import Dataset, DataLoader

nltk.download('punkt')
nltk.download('stopwords')

class SentimentDataset(Dataset):
    def __init__(self, sentiment_file, chinese_file, english_file):
        self.data = MSCTD_raw_data_preprocess(sentiment_file, chinese_file, english_file)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'sentiment': item['sentiment'],
            'chinese_text': item['chinese_text'],
            'english_text': item['english_text']
        }
    
def preprocess_chinese(text):
    tokens = jieba.cut(text)
    processed_text = ' '.join(tokens)
    return processed_text

def preprocess_english(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if word.isalnum()]
    processed_text = ' '.join(tokens)
    return processed_text


def MSCTD_raw_data_preprocess(sentiment_file, chinese_text_file, english_text_file):
    with open(sentiment_file, 'r', encoding='utf-8') as file:
        sentiments = file.readlines()

    with open(chinese_text_file, 'r', encoding='utf-8') as file:
        chinese_texts = file.readlines()

    with open(english_text_file, 'r', encoding='utf-8') as file:
        english_texts = file.readlines()
        
    if len(sentiments) != len(chinese_texts) and len(sentiments) != len(english_texts):
        raise ValueError("The number of lines in the sentiment file and the text file do not match.")

    processed_data = []
    for sentiment, chinese_text, english_text in zip(sentiments, chinese_texts, english_texts):
        processed_chinese_text = preprocess_chinese(chinese_text)
        processed_english_text = preprocess_english(english_text)
        processed_data.append({'sentiment': sentiment.strip(), 
                               'chinese_text': processed_chinese_text, 
                               'english_text': processed_english_text})

    return processed_data

def MSCTD_dataset(sentiment_train, chinese_train, english_train, 
                     sentiment_dev, chinese_dev, english_dev, 
                     sentiment_test, chinese_test, english_test, 
                     ):
    
    train_dataset = SentimentDataset(sentiment_train, chinese_train, english_train)
    dev_dataset = SentimentDataset(sentiment_dev, chinese_dev, english_dev)
    test_dataset = SentimentDataset(sentiment_test, chinese_test, english_test)

    return train_dataset, dev_dataset, test_dataset

if __name__ == "__main__":
    
    train_dataset, dev_dataset, test_dataset = MSCTD_dataset(
        'MSCTD_data/enzh/sentiment_train.txt', 'MSCTD_data/enzh/chinese_train_seg.txt', 'MSCTD_data/enzh/english_train.txt',
        'MSCTD_data/enzh/sentiment_dev.txt', 'MSCTD_data/enzh/chinese_dev_seg.txt', 'MSCTD_data/enzh/english_dev.txt',
        'MSCTD_data/enzh/sentiment_test.txt', 'MSCTD_data/enzh/chinese_test_seg.txt', 'MSCTD_data/enzh/english_test.txt'
    )
    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(train_loader)