import pandas as pd
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import torch
import pandas as pd
from torch.utils.data import Dataset
import torch.nn as nn

def calculate_accuracy(predicted, actual):
    """
    Calculate the accuracy of predictions.

    :param predicted: Predicted outputs from the model.
    :param actual: True labels.
    :return: Accuracy as a percentage.
    """
    predicted_softmax = torch.log_softmax(predicted, dim=1)
    _, predicted_tags = torch.max(predicted_softmax, dim=1)

    correct_predictions = (predicted_tags == actual).float()
    accuracy = correct_predictions.sum() / len(correct_predictions)

    accuracy = torch.round(accuracy * 100)

    return accuracy

class EnergyClassifierDataset(Dataset):
    """
    Dataset class for energy classification.
    """
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__ (self):
        return len(self.X_data)

class EnergyClassifierModel(nn.Module):
    """
    Neural network model for energy classification.
    """
    def __init__(self, num_features, num_classes):
        super(EnergyClassifierModel, self).__init__()

        self.st_layer = nn.Linear(num_features, 512)
        self.nd_layer = nn.Linear(512, 256)
        self.rd_layer = nn.Linear(256, 128)
        self.th_layer = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, num_classes)

        self.relu_activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=0.175)
        self.st_bn_layer = nn.BatchNorm1d(512)
        self.nd_bn_layer = nn.BatchNorm1d(256)
        self.rd_bn_layer = nn.BatchNorm1d(128)
        self.th_bn_layer = nn.BatchNorm1d(64)

    def forward(self, input_data):
        input_data = self.st_layer(input_data)
        input_data = self.st_bn_layer(input_data)
        input_data = self.relu_activation(input_data)

        input_data = self.nd_layer(input_data)
        input_data = self.nd_bn_layer(input_data)
        input_data = self.relu_activation(input_data)
        input_data = self.dropout_layer(input_data)

        input_data = self.rd_layer(input_data)
        input_data = self.rd_bn_layer(input_data)
        input_data = self.relu_activation(input_data)
        input_data = self.dropout_layer(input_data)

        input_data = self.th_layer(input_data)
        input_data = self.th_bn_layer(input_data)
        input_data = self.relu_activation(input_data)
        input_data = self.dropout_layer(input_data)
        input_data = self.output_layer(input_data)

        return input_data

def read_data(file_path):
    """
    Reads data from a CSV file.

    :param file_path: Path to the CSV file.
    :return: DataFrame with data from the CSV file.
    """
    data_frame = pd.read_csv(file_path)
    return data_frame

def partition_dataset(data, save_path):
    """
    Splits the dataset into training and validation sets.
    :param data: DataFrame containing the dataset.
    :param save_path: Path to save the validation dataset.
    :return: Training and validation feature and label arrays.
    """
    feature_columns = data.iloc[:, 1:-1]
    label_column = data.iloc[:, -1]
    features_train, features_val, labels_train, labels_val = train_test_split(feature_columns, label_column, test_size=0.2, shuffle=False)
    
    # Saving the validation dataset
    validation_data = pd.concat([features_val, labels_val], axis=1)
    validation_data.to_csv(save_path, index=False)

    return np.array(features_train), np.array(features_val), np.array(labels_train), np.array(labels_val)

def execute_training(features_train, labels_train, features_val, labels_val):
    """
    Executes the model training process.
    :param features_train: Training feature set.
    :param labels_train: Training label set.
    :param features_val: Validation feature set.
    :param labels_val: Validation label set.
    :return: Trained model.
    """
    Batch = 64

    epoc = 20
    nb_feat = 8
    cat = 8
    training_metrics = {'accuracy': [], 'loss': []}
    validation_metrics = {'accuracy': [], 'loss': []}


    dataset_train = EnergyClassifierDataset(torch.from_numpy(features_train).float(), torch.from_numpy(labels_train).long())
    dataset_val = EnergyClassifierDataset(torch.from_numpy(features_val).float(), torch.from_numpy(labels_val).long())
    
    loader_train = DataLoader(dataset_train, batch_size=Batch)
    loader_val = DataLoader(dataset_val, batch_size=1)

    energy_model = EnergyClassifierModel(nb_feat, cat)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(energy_model.parameters(), lr=0.002)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    energy_model.to(device)

    print("Starting training.")
    for epoch in range(1, epoc + 1):
        energy_model.train()
        train_loss, train_acc = 0, 0
        for inputs, targets in loader_train:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = energy_model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += calculate_accuracy(outputs, targets)

        energy_model.eval()

        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for inputs, targets in loader_val:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = energy_model(inputs)
                loss = loss_fn(outputs, targets)

                val_loss += loss.item()
                val_acc += calculate_accuracy(outputs, targets)

        # Update metrics
        training_metrics['loss'].append(train_loss / len(loader_train))
        training_metrics['accuracy'].append(train_acc / len(loader_train))
        validation_metrics['loss'].append(val_loss / len(loader_val))
        validation_metrics['accuracy'].append(val_acc / len(loader_val))

        # Print epoch statistics
        print("epoch : ",epoch)

    print("Training complete. Evaluating final accuracy...")
    final_train_accuracy = sum(training_metrics['accuracy'])/len(training_metrics['accuracy'])
    final_val_accuracy =sum(validation_metrics['accuracy'])/len(validation_metrics['accuracy'])


    print(f"Final Training Accuracy: {final_train_accuracy:.3f}")
    print(f"Final Validation Accuracy: {final_val_accuracy:.3f}")

    energy_model.eval()
    return energy_model

def store_trained_model(trained_model, path_to_model):
    """
    Stores the trained model at the specified path.
    :param trained_model: The model to be saved.
    :param path_to_model: Path where the model will be saved.
    """
    torch.save(trained_model.state_dict(), path_to_model)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Model training script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='../data/processed_data.csv', 
        help='Path to the processed data file to train the model'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='../models/model.pt', 
        help='Path to save the trained model'
    )
    return parser.parse_args()

def execute_main_process(data_file_path, output_model_path):
    """
    Main process for loading data, training the model, and saving the trained model.
    :param data_file_path: Path to the input data file.
    :param output_model_path: Path to save the trained model.
    """
    data_frame = read_data(data_file_path)
    features_train, features_val, labels_train, labels_val = partition_dataset(data_frame, "../data/test.csv") 
    trained_model = execute_training(features_train, labels_train, features_val, labels_val)
    store_trained_model(trained_model, output_model_path)

if __name__ == "__main__":
    args = parse_arguments()
    execute_main_process(args.input_file, args.model_file)
