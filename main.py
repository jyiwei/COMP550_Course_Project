from gensim.models import KeyedVectors
import numpy as np

from config import Config
from utils import *

from preprocess import *
from model.lstm import *

import matplotlib.pyplot as plt 

if __name__ == "__main__":

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed if torch.cuda.is_available() else 0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataloader, valid_dataloader, test_dataloader, vocab = pytorch_word2vec_dataloader()

    vocab_size = len(vocab)

    word2vec_model = load_word2vec_model(Config.word2vec_path)
    embedding_matrix = create_embedding_matrix(word2vec_model, vocab, Config.embedding_dim)

    model = LSTM_attention(vocab_size, 
                           Config.embedding_dim, 
                           embedding_matrix.clone().detach(), 
                           True, # update word2vec
                           Config.hidden_dim, 
                           Config.num_layers, 
                           Config.drop_keep_prob, 
                           Config.n_class, 
                           Config.bidirectional)
    model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
    num_epochs = Config.num_epochs

    train_losses = []
    valid_losses = []
    valid_accuracies = []

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        total_train_loss = 0
        for batch in train_dataloader:
            packed_sequences, labels = batch
            packed_sequences, labels = packed_sequences.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(packed_sequences)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)  

        # Validation Phase
        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            correct_predictions = 0
            total_samples = 0

            for batch in valid_dataloader:
                packed_sequences, labels = batch
                packed_sequences, labels = packed_sequences.to(device), labels.to(device)

                outputs = model(packed_sequences)
                loss = criterion(outputs, labels)

                total_valid_loss += loss.item()

                _, predicted = torch.max(outputs, 1)  
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

            avg_valid_loss = total_valid_loss / len(valid_dataloader)
            accuracy = (correct_predictions / total_samples) * 100.0  
            valid_losses.append(avg_valid_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, Accuracy: {accuracy:.2f}%')

        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)
        valid_accuracies.append(accuracy)
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(valid_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.figure(figsize=(10, 5))
    plt.plot(valid_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Validation Accuracy')

    plt.show()

    model.eval()
    total_test_loss = 0
    try:
        for batch in test_dataloader:
                packed_sequences, labels = batch
                packed_sequences, labels = packed_sequences.to(device), labels.to(device)

                outputs = model(packed_sequences)
                loss = criterion(outputs, labels)

                total_test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)  
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    avg_test_loss = total_test_loss / len(test_dataloader)
    test_accuracy = (correct_predictions / total_samples) * 100.0  

    print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
