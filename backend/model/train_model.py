import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib # For saving the model

def train_model(df):
    """Train a machine learning model on the given DataFrame."""

    # Separate target (y) and features (X)
    X = df.drop("tip", axis=1)
    y = df["tip"]

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=["category", "object"]).columns
    numerical_cols = X.select_dtypes(exclude=["category", "object"]).columns

    # Preprocessor: passthrough numericals, one-hot encode categoricals
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    # Create a pipeline: preprocessing → model
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
        ]
    )

    # Split data (⚠️ removed the extra comma here!)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit the model
    model.fit(X_train, y_train)

    return model
