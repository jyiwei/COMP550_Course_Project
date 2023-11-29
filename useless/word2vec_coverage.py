from gensim.models import KeyedVectors
from dataloader import *
import spacy

def calculate_model_coverage(dataset, word_vectors, verPP = 'chinese_text'):
    total_words = 0
    covered_words = 0

    for i in range(len(dataset)):
        data_entry = dataset[i]
        text = data_entry[verPP]
        print(text)
        for word in text.split(): 
            total_words += 1
            if word in word_vectors.key_to_index:  
                covered_words += 1

    coverage_percent = (covered_words / total_words) * 100 if total_words > 0 else 0
    return coverage_percent

if __name__ == "__main__":

    #50 vector size
    chinese_model_path = 'pretrained\wiki_word2vec_50.bin'
    chinese_word2vec = KeyedVectors.load_word2vec_format(chinese_model_path, binary=True) 
    
    english_model_path = 'pretrained\english_model.bin'
    english_word2vec = KeyedVectors.load_word2vec_format(english_model_path, binary=True) 
 

    #words_in_model = list(english_word2vec.key_to_index.keys())
    #print(words_in_model[:10])

    train_dataset, dev_dataset, test_dataset = MSCTD_dataset(
        'MSCTD_data/enzh/sentiment_train.txt', 'MSCTD_data/enzh/chinese_train_seg.txt', 'MSCTD_data/enzh/english_train.txt',
        'MSCTD_data/enzh/sentiment_dev.txt', 'MSCTD_data/enzh/chinese_dev_seg.txt', 'MSCTD_data/enzh/english_dev.txt',
        'MSCTD_data/enzh/sentiment_test.txt', 'MSCTD_data/enzh/chinese_test_seg.txt', 'MSCTD_data/enzh/english_test.txt'
    )


    coverage_train = calculate_model_coverage(train_dataset, chinese_word2vec, verPP = 'chinese_text')
    print(f"Model coverage for the Chinese Train: {coverage_train}%")

    coverage_dev = calculate_model_coverage(dev_dataset, chinese_word2vec, verPP = 'chinese_text')
    print(f"Model coverage for the Chinese Dev: {coverage_dev}%")

    coverage_test = calculate_model_coverage(test_dataset, chinese_word2vec, verPP = 'chinese_text')
    print(f"Model coverage for the Chinese Test: {coverage_test}%")
    ################################################################################################
    
    ################################################################################################
    coverage_train = calculate_model_coverage(train_dataset, english_word2vec, verPP = 'english_text')
    print(f"Model coverage for the English Train: {coverage_train}%")

    coverage_dev = calculate_model_coverage(dev_dataset, english_word2vec, verPP = 'english_text')
    print(f"Model coverage for the English Dev: {coverage_dev}%")

    coverage_test = calculate_model_coverage(test_dataset, english_word2vec, verPP = 'english_text')
    print(f"Model coverage for the English Test: {coverage_test}%")