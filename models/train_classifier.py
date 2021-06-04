# import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])

import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report , f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import pickle


def load_data(database_filepath):

    '''
    Function to retreive data from sql database and return a pandas df

    Args: 
    database_filepath: str. Filepath of database

    Returns: Dataframe which will be returned under the variable name "df"

    '''
    engine = create_engine('sqlite:///' + database_filepath) #sqlite:///DisasterMsgs.db
    df = pd.read_sql_query('''SELECT * FROM disastermsgs''', engine)

    return  df

def prep_data(df):

    '''
    Function to prepare the dataframe loaded to be used in machine learning models
    Args:
    df: dataframe loaded with the load_data function
    Returns: Features X & target Y along with target columns names target_names
    '''
    
    df = df.dropna() # dropping NAs
    
    Y = df.iloc[:, 4:]
    
    X = df['message'].values # Features
    Y = df.iloc[:, 4:] # Target
    target_names = Y.columns.values

    return X, Y, target_names


def tokenize(text):
    
    '''
    Function to clean the text data  and apply tokenize and lemmatizer function
    
    Args: 
    text: text data
    Returns cleaned tokenized text as a list object
    '''

    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():

    '''
    Function to build the model, create pipeline as well as perform GridSearchCV
    Input: N/A
    Output: Returns the model
    '''


    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs = -1)))
    ])


    parameters = {'clf__estimator__n_estimators': [100, 200]   
             }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):

    '''
    Function to evaluate a model and return the classification and accuracy score.
    Inputs: Model, X_test, y_test, target_names
    Output: Prints the Classification report as pandas dataframe
    '''

    y_pred = model.predict(X_test)

    f1_scores = []
    for ind, cat in enumerate(Y_test):
        print('Class - {}'.format(cat))
        print(classification_report(Y_test.values[ind], y_pred[ind], zero_division = 1))
    
        f1_scores.append(f1_score(Y_test.values[ind], y_pred[ind], zero_division = 1))
  
    print('Trained Model\nMinimum f1 score - {}\nBest f1 score - {}\nMean f1 score - {}'.format(min(f1_scores), max(f1_scores), round(sum(f1_scores)/len(f1_scores), 3)))
    print("\nBest Parameters:", model.best_params_)


def save_model(model, model_filepath):

    '''
    Function to save the model as pickle file in the directory
    Input: model and the file path to save the model
    Output: save the model as pickle file in the given filepath 
    '''

    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        df = load_data(database_filepath)
        X, Y, category_names = prep_data(df)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state = 42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()