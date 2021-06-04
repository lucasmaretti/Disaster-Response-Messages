import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(path_messages, path_categories):

    '''
    Function to load "messages" and "categories" csv files as pandas dataframe.
    
    Args:
    path_messages: str. messages.csv file path name as string
    path_categories: str. categories.csv file path name as string

    Returns: each file loaded as a pandas dataframe
    
    '''
   #Read csv file and load in the variable as dataframe
    messages = pd.read_csv(path_messages)
    categories = pd.read_csv(path_categories)

    return messages, categories

def clean_data(messages, categories):

    '''
    Function to clean the dataframe inorder to be compatible with Machine Learning steps further.
   
    More details on all steps are described inline comments within the function
    
    Args: 
    messages: Pandas dataframe.
    categories: Pandas dataframe

    Returns: cleaned and formatted dataframe which will be returned under the variable name "df"
    '''

    # merging categories and messages dataframes
    df = pd.merge(messages, categories, how = 'left', on = 'id') # merging categories and messages dataframes

    # creating column headers for each category that will be splitted and then transforming to dummy
    categories = categories.categories.str.split(pat = ';', expand = True)
    row = categories[:1]
    category_colnames = row.iloc[0].apply(lambda x: x[:-2]).tolist()
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    #updating df with new categories dataframe

    #related column contains some values of '2', re-assign them as '1'
    categories.loc[categories['related']==2,'related']=1
    
    #'child_alone' category is empty, we would not be able to train on it, drop it
    categories = categories.drop(['child_alone'], axis = 1)
    df = df.drop(columns = ['categories'])
    df = pd.concat([df,categories], axis = 1)

    # drop duplicates
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_name):

    '''
    Function to save the cleaned dataframe into a sql database table with name 'messages'

    Input: 
    df: pandas dataframe from load_data function
    database_name: str. Name of the database to be created

    Returns: SQLite Database with name 'database_name' and table 'messages' with df loaded into.
    '''

    engine = create_engine('sqlite:///' + database_name)
    df.to_sql('disastermsgs', engine, index=False, if_exists = 'replace')

def main():
    if len(sys.argv) == 4:

        path_messages, path_categories, database_name = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(path_messages, path_categories))
        messages, categories = load_data(path_messages, path_categories)

        print('Cleaning data...')
        df = clean_data(messages, categories)
        
        print('Saving data...\n    DATABASE: {}'.format(database_name))
        save_data(df, database_name)
        
        print('Cleaned data saved to database.')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively and'\
              'name of the database to save the cleaned data '\
              'to as the third argument.')


if __name__ == '__main__':
    main()
