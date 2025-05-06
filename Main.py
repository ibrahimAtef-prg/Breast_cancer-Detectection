
# Loading the required libraries

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report





def main():

    # Load the dataset

    data = pd.read_csv('BreastCancer_DS.csv')

    # Display the first few rows of the dataset
    print(data.head())
    print(data.info())
    print(data.describe())

    # preprocess the data

    # check for missing values
    print(data.isnull().sum())

    # drop the 'Unnamed: 32' column and 'id' column
    data.drop(['id'], axis=1, inplace=True)

    # change the 'diagnosis' column to binary values
    data['diagnosis'] = data['diagnosis'].map({'M': 1,
                                                'B': 0}) # 1 for malignant, 0 for benign
    print(data['diagnosis'].value_counts())

    # we can also use OneHotEncoder for categorical variables
    
    # Split the data into features and target variable

    X = data.drop(['diagnosis'], axis=1)

    Y = data['diagnosis']

    # Split the data into training and testing sets
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Create a Logistic Regression model
    
    model = LogisticRegression()

    # train model

    model.fit(X_train, Y_train)

    # make classifications
    y_class = model.predict(X_test)

    # evaluate the model
    accuracy = accuracy_score(Y_test, y_class)
    print(f'Accuracy: {accuracy * 100:.2f}%')


    print('Confusion Matrix:')
    print(np.array(confusion_matrix(Y_test, y_class)))

    
    print('Classification Report:')
    print(classification_report(Y_test, y_class))

main()