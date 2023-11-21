import pandas as pd
import numpy as np
import argparse
import torch
from model_training import EnergyClassifierModel
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
from model_training import EnergyClassifierDataset,partition_dataset


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def load_model(model_path):
    model_state_dict = torch.load(model_path)
    model = EnergyClassifierModel(8,8)
    model.load_state_dict(model_state_dict)
    return model

def make_predictions(df, model):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    df = df.iloc[:,:-1]
    model.to(device)
    model.eval()
    
    input_data = torch.tensor(df.values, dtype=torch.float32) 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_data = input_data.to(device)

    with torch.no_grad():
        output = model(input_data)
    
    probabilities = torch.log_softmax(output, dim = 1)

    _, predicted_classes = torch.max(probabilities, 1)

    predicted_classes = predicted_classes.cpu().numpy().tolist()
    
    return predicted_classes

def save_predictions(predictions, predictions_file):
    json_data = {"target": {}}
    for i, num in enumerate(predictions):
        if i == 442:
            break
        json_data["target"][str(i)] = num
    print("predictions.json created")
    with open(predictions_file, 'w') as json_file:
        json.dump(json_data, json_file, indent=2)
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Prediction script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='../data/test.csv',
        help='Path to the test data file to make predictions'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='../models/model.pt',
        help='Path to the trained model file'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='../predictions/predictions.json',
        help='Path to save the predictions'
    )
    return parser.parse_args()

def main(input_file, model_file, output_file):
    df = load_data(input_file)
    model = load_model(model_file)
   
    predictions = make_predictions(df, model)
    save_predictions(predictions, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file, args.output_file)