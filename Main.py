
# Loading the required libraries

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report





def main():
    # Load the dataset
    data = pd.read_csv('C:\\Users\\pc\\OneDrive\\Desktop\\ai\\college\\Sem.2\\Machine Learning\\PRj.1\\BreastCancer_DS.csv')


    # preprocessing the dataset
    df = pd.DataFrame(data)
    print(df.info())
    print(df.describe())

    # Display the first few rows of the dataset
    print(df.head())

        # Check for missing values
    print("Missing values in each column:")
    print(df.isnull().sum())

    # Decode the target variable 'diagnosis'
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0}) # we can use the one hot encoding as well but mapping is more efficient now

    # Split dataset into features and target variable
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']

    # training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Using Logistic Regression model
    model = LogisticRegression()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy*100:.2f}")

    # Display confusion matrix and classification report to measure the performance of the model
    matrix = np.array(confusion_matrix(y_test, y_pred))
    print("Confusion Matrix:")
    print(matrix)
    print("Classification Report:",classification_report(y_test, y_pred))

main()