import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# from sklearn.externals import joblib
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
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals

    # First visualisation - distribution of message genres - direct, news, social
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Second visualisation - total categories
    total_category = df.drop(columns=['id', 'message', 'original', 'genre'], axis=1).sum().sort_values(ascending=False).head(10)

    # Third visualisation - histogram of text length
    df['text length'] = df['message'].apply(lambda x: len(x.split()))
    histogram = df[df['text length'] < 100].groupby('text length').count()['id']

    # create visuals
    graphs = [
        # 1st Visualisation
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
                }
            }
        },
        # 2nd Visualisation
        {
            'data': [
                Bar(
                    x=total_category.index,
                    y=total_category.values
                )
            ],

            'layout': {
                'title': 'Total Messages Count for Top 10 Categories',
                'yaxis': {
                    'title': "Total"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },

        # 3rd Visualisation
        {
            'data': [
                Bar(
                    x=histogram.index,
                    y=histogram.values
                )
            ],

            'layout': {
                'title': 'Distribution of Messages Length',
                'yaxis': {
                    'title': "Total Messages"
                },
                'xaxis': {
                    'title': "Total Words"
                }
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