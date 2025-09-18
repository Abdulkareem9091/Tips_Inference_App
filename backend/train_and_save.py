"""The purpose of theis file is to load the data from load_data.py 
and load the trained model from train_model.py and save it to file """

import joblib

from data.load_data import load_data
from model.train_model import train_model

def train_and_save_model():
    """Load data, train the model, and save it to a file. """
    #Load the data
    df = load_data() #This function provides the dataframe

    # Train the model
    model = train_model(df) #This function provides the trined model

    # save the trained model to a file
    joblib.dump(model, "random_forest_model.pkl")
    print("Model saved to random_forest_model.pkl")

if __name__ == "__main__":
    train_and_save_model()