""" from flask import Flask, request, jsonify
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re

app = Flask(__name__)

def file_to_wordcloud(file_content):
    sid = SentimentIntensityAnalyzer()

    # Calculate the sentiment scores for each word in the file
    sentiment_scores = {}
    words = re.findall(r'\b\w+\b', file_content)
    for word in words:
        ss = sid.polarity_scores(word)
        sentiment_scores[word] = ss['compound']

    # Sort the words based on their sentiment scores
    sorted_words = sorted(sentiment_scores.items(), key=lambda x: x[1], reverse=True)

    # Generate the word cloud
    top_5_words = ' '.join([word for word, score in sorted_words[:5]])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(top_5_words)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url

@app.route('/')
def hello_world():
   return 'Hello World'

@app.route('/api/convert', methods=['POST'])
def convert_to_wordcloud():
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    file_content = file.read().decode('utf-8')
    wordcloud_url = file_to_wordcloud(file_content)
    return jsonify({'wordcloud_url': wordcloud_url})

if __name__ == '__main__':
    app.run(debug=True) """


from flask import Flask, request, jsonify
from wordcloud import WordCloud
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
def file_to_wordcloud_pos(file_content):
    sid = SentimentIntensityAnalyzer()

    # Calculate the sentiment scores for each word in the file
    sentiment_scores = {}
    words = re.findall(r'\b\w+\b', file_content)
    
    for word in words:
        ss = sid.polarity_scores(word)
        sentiment_scores[word] = ss['compound']

    # Sort the words based on their sentiment scores
    sorted_words = sorted(sentiment_scores.items(), key=lambda x: x[1], reverse=True)

    # Generate the word cloud
    top_5_words = ' '.join([word for word, score in sorted_words[:5]])
    wordcloud = WordCloud(width=390, height=307, background_color='black').generate(top_5_words)
    plt.figure( figsize=(20,10), facecolor='k' )
    plt.tight_layout(pad=0)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url

def file_to_wordcloud_neg(file_content):
    sid = SentimentIntensityAnalyzer()

    # Calculate the sentiment scores for each word in the file
    sentiment_scores = {}
    words = re.findall(r'\b\w+\b', file_content)
    
    for word in words:
        ss = sid.polarity_scores(word)
        sentiment_scores[word] = ss['compound']

    # Sort the words based on their sentiment scores
    sorted_words = sorted(sentiment_scores.items(), key=lambda x: x[1], reverse=True)

    # Generate the word cloud
    top_5_words = ' '.join([word for word, score in sorted_words[-5:]])
    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(top_5_words)

    plt.figure( figsize=(20,10), facecolor='k' )
    plt.tight_layout(pad=0)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url

@app.route('/api/convert/pos', methods=['POST'])

def convert_to_wordcloud_pos():
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    file_content = file.read().decode('utf-8')
    wordcloud_url = file_to_wordcloud_pos(file_content)
    return jsonify({'wordcloud_url': wordcloud_url})
@app.route('/api/convert/neg', methods=['POST'])
def convert_to_wordcloud_neg():
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    file_content = file.read().decode('utf-8')
    wordcloud_url = file_to_wordcloud_neg(file_content)
    return jsonify({'wordcloud_url': wordcloud_url})

if __name__ == '__main__':
    app.run(debug=True)