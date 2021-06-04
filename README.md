# Disaster Response Pipeline Project

## Table of Contents

1. [Project Motivation](#motivation)
2. [Requirements](#requirements)
3. [File Structure](#files)
4. [Installation](#installation)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Motivation<a name="motivation"></a>
This project aims to put into practice software engineering and data engineering skills based on the Data Scientist Nanodegree by Udacity. The was to analyze disaster data from real life occurances provided by [Figure Eight](https://www.figure-eight.com/) and build a web app that, based on a machine learning model that was trained on messages provided by the company, categorize new messages according to 36 previously defined categories. The steps into doing this were:

1. Construct an ETL (Extract, Transform, Load) pipeline.

2. Train a supervised learning model (Random Forest Classifier).

3. Build a web app using Flask that runs locally and can be hosted on a platform such as Heroku to receive new messages and categorizes them.


## Requirements <a name="requirements"></a>
All the libraries required for the app to run are listed in the requirements.txt file

## File structure <a name="files"></a>

'data' folder:
-  process_data.py >>> python script that generates DisasterResponse database from csv files (ETL pipeline)
- categories.csv and messages.csv >>> original message and categories files used to train the model and build the SQLite database

'models' folder:
- train_classifier.py >>> python script that builds a machine learning pipeline to train the Random forest classifier model
- classifier.pkl >>> pickle file of the trained model

'app' folder:
- run.py >>> python script that runs the app locally
- templates >>> html files to render the app

### Download and Installation
```console
foo@bar:~ $ git clone https://github.com/lucasmaretti/Disaster-Response-Messages.git
foo@bar:~ $ cd Disaster-Response-Messages
foo@bar:Disaster-Response-Messages $  
```
While in the project's root directory `Disaster-Response-Messages` run the ETL pipeline that cleans and stores data in database.
```console
foo@bar:disaster-response-pipeline $ python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db
```
Next, run the ML pipeline that trains the classifier and saves it.
```console
foo@bar:disaster-response-pipeline $ python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

Next, change directory into the `app` directory and run the Python file `run.py`.
```console
foo@bar:disaster-response-pipeline $ cd app
foo@bar:app $ python run.py
```

Finally, go to http://0.0.0.0:3001/ or http://localhost:3001/ in your web-browser.

Type a message input box and click on the `Classify Message` button to see how the various categories that your message falls into.


## Licensing and Acknowledgements<a name="licensing"></a>

Big credit goes to [Figure Eight](https://appen.com/) for the relabelling the datasets and also to the teaching staffs at [Udacity](https://www.udacity.com/). All files included in this repository are free to use.

