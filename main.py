from gensim.models import KeyedVectors
import numpy as np

from utils import EarlyStopping 
from utils import load_word2vec_model, create_embedding_matrix, check_word2vec_coverage  
from preprocess import pytorch_word2vec_dataloader, pytorch_bag_of_words_dataloader
from model.lstm import LSTM_attention, LSTMModel
from model.logistic_regression import LogisticRegression

SEED = 42
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED if torch.cuda.is_available() else 0)

    if Config.model_name == 'BiLSTM_A' or Config.model_name == 'LSTM':
        train_dataloader, valid_dataloader, test_dataloader, vocab = pytorch_word2vec_dataloader()
    elif Config.model_name == 'LR':
        train_dataloader, valid_dataloader, test_dataloader, vocab = pytorch_bag_of_words_dataloader()

    vocab_size = len(vocab)
    print(f"Vocab size is {vocab_size}")

    word2vec_model = load_word2vec_model(Config.word2vec_path)
    embedding_matrix = create_embedding_matrix(word2vec_model, vocab, Config.embedding_dim)

    covered, oov = check_word2vec_coverage(vocab, word2vec_model)
    #print(oov)

    if Config.model_name == 'BiLSTM_A':
        model = LSTM_attention(vocab_size, 
                            Config.embedding_dim, 
                            Config.hidden_dim, 
                            Config.num_layers, 
                            Config.drop_keep_prob, 
                            Config.n_class, 
                            Config.bidirectional,
                            Config.use_pretrained
                            #    embedding_matrix.clone 
                            #    True, # update word2vec
                            )
    elif Config.model_name == 'LR':
        model = LogisticRegression(vocab_size, Config.n_class)
    elif Config.model_name == 'LSTM':
        model = LSTMModel(vocab_size, 
                          Config.embedding_dim, 
                          Config.hidden_dim, 
                          Config.drop_keep_prob, 
                          Config.n_class)

    model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
    num_epochs = Config.num_epochs
    early_stopping = EarlyStopping(patience=Config.patience, delta=0.001)

    train_losses = []
    valid_losses = []
    valid_accuracies = []

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        total_train_loss = 0
        correct_predictions = 0
        total_samples = 0

        for batch in train_dataloader:
            (padded_sequence, labels), seq_lengths = batch
            padded_sequence, labels = padded_sequence.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(padded_sequence, seq_lengths)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # Calculate the number of correct predictions
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Calculate training accuracy for this epoch
        training_accuracy = correct_predictions / total_samples * 100.0  



        # Validation Phase
        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            correct_predictions = 0
            total_samples = 0

            for batch in valid_dataloader:
                (padded_sequence, labels), seq_lengths = batch
                padded_sequence, labels = padded_sequence.to(device), labels.to(device)

                outputs = model(padded_sequence, seq_lengths)
                loss = criterion(outputs, labels)

                total_valid_loss += loss.item()

                _, predicted = torch.max(outputs, 1)  
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        avg_valid_loss = total_valid_loss / len(valid_dataloader)
        valid_accuracy = (correct_predictions / total_samples) * 100.0  
        valid_losses.append(avg_valid_loss)

        early_stopping(avg_valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Train Accuracy: {training_accuracy:.2f}%, Valid Loss: {avg_valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.2f}%')

        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)
        valid_accuracies.append(valid_accuracy)
    
    if Config.plot_graph:
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

    model_save_path = os.path.join(Config.saved_model_path, f"{Config.model_name}_{Config.mode}_{Config.lr}_{epoch}_{Config.num_layers}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")
    
    model.eval()
    total_test_loss = 0
    all_test_labels = []
    all_test_predictions = []
    try:
        for batch in test_dataloader:
                (padded_sequence, labels), seq_lengths = batch
                padded_sequence, labels = padded_sequence.to(device), labels.to(device)

                outputs = model(padded_sequence, seq_lengths)
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
