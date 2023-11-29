

class Config:
    update_w2v = True  
    vocab_size = 68419  
    n_class = 2  
    max_sen_len = 150 
    embedding_dim = 50  
    batch_size = 256  
    hidden_dim = 128  
    num_epochs = 100  
    lr = 0.0001  
    drop_keep_prob = 0.2  
    num_layers = 4  
    bidirectional = True  

    #pretrained_model
    word2vec_path = 'pretrained\wiki_word2vec_50.bin'

    # Path for dataset
    stopword_path = 'data\stopword.txt'
    dataset_path = 'data\online_shopping_10_cats.csv'
