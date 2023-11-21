#!/bin/bash



# Run data_ingestion.py
echo "Starting data ingestion..."
python ../src/data_ingestion.py 

# Run data_processing.py
echo "Starting data processing..."
python ../src/data_processing.py 

# Run model_training.py
echo "Starting model training..."
python ../src/model_training.py 

# Run model_prediction.py
echo "Starting prediction..."
python ../src/model_prediction.py

echo "Pipeline completed. ETL, Model.pt, prediciton.json done"
