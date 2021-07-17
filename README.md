# Disaster Response Pipeline Project

## Introduction
This project builds a web app which classifiies a message sent during a disaster.  The project is split into three main steps:

### 1. ETL process
We prepare the data for analysis by cleaning and merging the available datasets, namely `disaster_messages.csv` and `disaster_categories.csv`  

These steps are performed in the Python script `data/process_data.py`

### 2. Training the model
Using scikit-learn, we train the classifier to classify a message.

This is performed using the file `model/train_classifier.py`

### 3. Run the web app

Using flask, we run the model and deploy in the website.


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
