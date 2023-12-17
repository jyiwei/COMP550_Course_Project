import os
import csv
import argparse

import torch
import torch.nn as nn
import numpy as np
from config import Config
import matplotlib.pyplot as plt 
from gensim.models import KeyedVectors
from sklearn.metrics import f1_score, roc_auc_score

from utils import EarlyStopping
from utils import load_word2vec_model, create_embedding_matrix, check_word2vec_coverage  
from preprocess import pytorch_word2vec_dataloader, pytorch_bag_of_words_dataloader
from model.lstm import LSTM_attention, LSTM_Model
from model.logistic_regression import LogisticRegression

SEED = 42
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(description="Run a machine learning model.")
    parser.add_argument("--model_name", type=str, choices=['LSTM', 'LSTM_A', 'LR'], required=True, help="Type of model to use (LSTM or LSTM_A or LR)")
    parser.add_argument("--mode", type=str, choices=['word', 'character', 'pingyin'], required=True, help="Granularity of chinese text feature")
    parser.add_argument("--lr", type=float, default= 0.0003, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of Epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    
    parser.add_argument("--no_stopwords", action="store_false", help="Do not use stopwords")
    parser.add_argument("--use_w2v", action="store_true", help="Use w2v")
    
    parser.add_argument("--embedding_dim", type=int, default=300, help="embedding dimension of LSTM")
    parser.add_argument("--hidden_dim", type=int, default=128, help="hidden dimension of LSTM")
    parser.add_argument("--drop_keep_prob", type=float, default=0.2, help="drop probability of drop out")
    parser.add_argument("--num_layers", type=int, default=2, help="drop probability of drop out")
    parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional LSTM")
    parser.add_argument("--n_classes", type=int, default=2, help="Number of classes, e.g. binary classification: 2")

    args = parser.parse_args()

    model_type = 'LSTM' if args.model_name == 'LSTM' or args.model_name == 'LSTM_A' else 'LR'    

    # Load the default configuration
    config_dir = 'configs/'
    config_file = config_dir+'lstm_default_config.json' if model_type == 'LSTM' else config_dir+'lr_default_config.json'
    config = Config(config_file)

    # Override parameters from command line args
    config.batch_size = args.batch_size
    config.mode = args.mode
    config.n_classes = args.n_classes
    config.use_stopwords = args.no_stopwords
    
    if model_type == 'LSTM':
        config.use_w2v = args.use_w2v 
        config.embedding_dim = args.embedding_dim
        config.hidden_dim = args.hidden_dim
        config.drop_keep_prob = args.drop_keep_prob
        config.num_layers = args.num_layers
        config.bidirectional = args.bidirectional

    train_dataloader, valid_dataloader, test_dataloader, vocab = pytorch_word2vec_dataloader(config) if model_type == 'LSTM' else pytorch_bag_of_words_dataloader(config)

    vocab_size = len(vocab)
    print(f"Vocab size is {vocab_size}")

    embedding_matrix = None
    if model_type == 'LSTM' and config.use_w2v:
        word2vec_model = load_word2vec_model(config.word2vec_path)
        embedding_matrix = create_embedding_matrix(word2vec_model, vocab, config.embedding_dim)

        covered, oov = check_word2vec_coverage(vocab, word2vec_model)
        print(oov)
    pretrained_weight = embedding_matrix.clone() if embedding_matrix is not None else None

    if args.model_name == 'LSTM':
        model = LSTM_Model(vocab_size, 
                           config.embedding_dim, 
                           config.hidden_dim, 
                           config.num_layers, 
                           config.drop_keep_prob, 
                           config.n_classes, 
                           config.bidirectional,
                           pretrained_weight,
                           config.update_w2v
                          )
    elif args.model_name == 'LSTM_A':
        model = LSTM_attention(vocab_size, 
                               config.embedding_dim, 
                               config.hidden_dim, 
                               config.num_layers, 
                               config.drop_keep_prob, 
                               config.n_classes, 
                               config.bidirectional,
                               pretrained_weight,
                               config.update_w2v
                              )
    elif args.model_name == 'LR':
        model = LogisticRegression(vocab_size, config.n_classes)

    model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    num_epochs = args.num_epochs
    early_stopping = EarlyStopping(patience=args.patience, delta=0.001)

    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        total_train_loss = 0
        correct_predictions = 0
        total_samples = 0

        for batch in train_dataloader:
            padded_sequence, labels = batch
            padded_sequence, labels = padded_sequence.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(padded_sequence)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # Calculate the number of correct predictions
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        avg_train_loss = total_train_loss / len(train_dataloader)

        # Calculate training accuracy for this epoch
        training_accuracy = correct_predictions / total_samples * 100.0  



        # Validation Phase
        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            correct_predictions = 0
            total_samples = 0

            for batch in valid_dataloader:
                padded_sequence, labels = batch
                padded_sequence, labels = padded_sequence.to(device), labels.to(device)

                outputs = model(padded_sequence)
                loss = criterion(outputs, labels)

                total_valid_loss += loss.item()

                _, predicted = torch.max(outputs, 1)  
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        avg_valid_loss = total_valid_loss / len(valid_dataloader)
        valid_accuracy = (correct_predictions / total_samples) * 100.0  

        early_stopping(avg_valid_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Train Accuracy: {training_accuracy:.2f}%, Valid Loss: {avg_valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.2f}%')

        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)
        train_accuracies.append(training_accuracy)
        valid_accuracies.append(valid_accuracy)
    
    if config.plot_graph:
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

    config_name = ""
    if model_type == 'LSTM':
        config_name = f"{args.model_name}_{args.mode}_w2v:{config.use_w2v}_bi:{config.bidirectional}_layer:{config.num_layers}_emb:{config.embedding_dim}_hid:{config.hidden_dim}_{epoch}"
    elif model_type == 'LR':
        config_name = f"{args.model_name}_{args.mode}_{epoch}"

    if config.saved_model_path is not None:
        print(f'Saving model to {config.saved_model_path}...')
        os.makedirs(config.saved_model_path, exist_ok=True)
        model_save_path = os.path.join(config.saved_model_path, config_name+".pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved at {model_save_path}")
    
    model.eval()
    total_test_loss = 0
    all_test_labels = []
    all_test_predictions = []
    try:
        for batch in test_dataloader:
                padded_sequence, labels = batch
                padded_sequence, labels = padded_sequence.to(device), labels.to(device)

                outputs = model(padded_sequence)
                loss = criterion(outputs, labels)

                total_test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)  
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                all_test_labels.extend(labels.cpu().numpy())
                all_test_predictions.extend(predicted.cpu().numpy())

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    avg_test_loss = total_test_loss / len(test_dataloader)
    test_accuracy = (correct_predictions / total_samples) * 100.0  

    test_f1_score = f1_score(all_test_labels, all_test_predictions, average='weighted')
    test_auc_roc = roc_auc_score(all_test_labels, all_test_predictions)
    print('-----------------------------------------------------------------------------------------------------------------')
    print(f'Test Accuracy: {test_accuracy:.2f}%, Test F1-Score: {test_f1_score:.4f}, Test AUC-ROC: {test_auc_roc:.4f}')
    print('-----------------------------------------------------------------------------------------------------------------')

    #logging
    if config.log_path is not None:
        print(f'Writing training logs to {config.log_path}...')
        os.makedirs(config.log_path, exist_ok=True)
        with open(os.path.join(config.log_path, config_name + '_train_results.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train Loss", "Valid Loss", "Train Accuracy", "Valid Accuracy"])
            for epoch in range(len(train_losses)):
                writer.writerow([epoch+1, train_losses[epoch], valid_losses[epoch], train_accuracies[epoch], valid_accuracies[epoch]])
        with open(os.path.join(config.log_path, config_name + '_test_results.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Test Loss", "Test Accuracy", "Test F1-Score", "Test AUC-ROC"])
            writer.writerow([avg_test_loss, test_accuracy, test_f1_score, test_auc_roc])
        print(f"Logged")


if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED if torch.cuda.is_available() else 0)

    main()




