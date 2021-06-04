import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar

#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disastermsgs', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    cat_df = df.drop(columns=['id', 'message', 'original', 'genre'])

    cat_count = cat_df.sum().sort_values(ascending=False)[:10]

    categories = cat_count.index

    coln_labels = df.drop(columns=['id', 'message', 'original', 'genre']).sum().sort_values(ascending=False).index

    df_genre_grps = df.groupby('genre')[coln_labels].sum().reset_index()


    df_genre = df_genre_grps.drop(columns=['genre']).rename(index={0 : 'direct', 1:'news', 2:'social'})
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }, 
                'template': "plotly_dark"
            }
        }, 
        
        {
            'data': [
                Bar(
                    x=categories,
                    y=cat_count
                )
            ],

            'layout': {
                'title': 'Top 10 Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message Categories"
                }, 
                'template': 'plotly_dark'
            }
        },
        
        {
    		'data': [
    					{
    						'name': 'direct',
              				'type': 'bar',
              				'x': coln_labels[:10],
              				'xaxis': 'x',
              				'y': df_genre.iloc[0],
              				'yaxis': 'y'
              			},
             			{
             				'name': 'news',
              				'type': 'bar',
              				'x': coln_labels[:10],
              				'xaxis': 'x2',
              				'y': df_genre.iloc[1],
              				'yaxis': 'y2'
              			},
             			{
             				'name': 'social',
              				'type': 'bar',
              				'x': coln_labels[:10],
              				'xaxis': 'x3',
              				'y': df_genre.iloc[2],
              				'yaxis': 'y3'
              			}
              		], 
              		
    		'layout': {
    					'template': 'plotly_dark',
               			'title': {'text': 'Top 10 Messages Stratified by Genre'},
               			'xaxis': {'anchor': 'y', 'domain': [0,0.25]},
               			'xaxis2': {'anchor': 'y2', 'domain': [0.35,0.65]},
               			'xaxis3': {'anchor': 'y3', 'domain': [0.75,1.0]},
               			'yaxis': {'anchor': 'x', 'domain': [0,1.0]},
               			'yaxis2': {'anchor': 'x2', 'domain': [0,1.0]},
               			'yaxis3': {'anchor': 'x3', 'domain': [0,1.0]}
               		}
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()