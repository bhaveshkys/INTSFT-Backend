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


from flask import Flask, request, jsonify,send_file
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
from collections import Counter
import io
import os
from operator import itemgetter
import numpy as np
from nltk.tokenize import word_tokenize
from pydub import AudioSegment
from pydub.utils import which
import speech_recognition as sr
AudioSegment.converter = which("ffmpeg")
app = Flask(__name__)
CORS(app)
temp_dir = os.path.abspath('temp')
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
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

@app.route('/api/plot', methods=['POST'])
def plot():
    # Retrieve text from request
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    text = file.read().decode('utf-8')

    # Text cleaning and word tokenization
    words = [word.lower() for word in word_tokenize(text)]
    words = [word for word in words if word.isalpha()]
    freq = Counter(words)
    sorted_freq = sorted(freq.items(), key=itemgetter(1), reverse=True)
    top_5_words = sorted_freq[:5]

    # Prepare data for plotting
    words_list = [word[0] for word in top_5_words]
    frequencies_list = [word[1] for word in top_5_words]

    # Plot
    fig = plt.figure(figsize=(10,5))
    plt.bar(words_list, frequencies_list)
    plt.title('Top 5 Frequently Used Words')
    plt.xlabel('Words')
    plt.ylabel('Frequency')

    # Save plot to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)

    # Convert plot to base64 encoded string
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')

    # Create response
    response = {
        'plot': plot_base64,
        'top_5_words': top_5_words
    }

    return jsonify(response)


def save_chunks_as_wav_files(chunks, output_folder):
    
    for i, chunk in enumerate(chunks):
        
        output_file_path = f"{output_folder}/chunk_{i}.wav"
        chunk.export(output_file_path, format="wav")

def split_audio_file(file_path, chunk_length_in_seconds):
    
    audio = AudioSegment.from_wav(file_path)
    chunks = []
    
    for i in range(0, len(audio), chunk_length_in_seconds * 1000):
        
        chunk = audio[i:i + chunk_length_in_seconds * 1000]
        chunks.append(chunk)
        
    return chunks

def speech_to_text(file_path):
    
    # Load the audio file
    audio = AudioSegment.from_wav(file_path)

    # Convert the audio file to the correct format for speech recognition
    audio.export("temp.wav", format="wav")

    # Use the speech recognition library to convert the speech in the audio file to text
    recognizer = sr.Recognizer()
    
    """ with sr.AudioFile("temp.wav") as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
    recognizer = sr.Recognizer() """
    
    try:
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_sphinx(audio_data)
            return text
    except sr.UnknownValueError:
        print("Unable to recognize speech in this chunk. Skipping to the next one.")
        return ""  # Return an empty string instead of raising the error

    return text

@app.route('/upload', methods=['POST'])
def upload_file():
    
    # Get the audio file from the request
    audio_file = request.files['audio']

    # Save the audio file temporarily
    temp_file_path = 'temp.mp3'
    audio_file.save(temp_file_path)

    # Convert the MP3 file to a WAV file
    audio = AudioSegment.from_file(temp_file_path)
    audio.export("temp.wav", format="wav")

    # Split the audio file into chunks
    chunk_length_in_seconds = 10
    chunks = split_audio_file("temp.wav", chunk_length_in_seconds)
    save_chunks_as_wav_files(chunks, temp_dir)

    # ... (The rest of the code in the previous steps goes here) ...
    # ... (The rest of the code in the previous steps goes here) ...

    # Combine all the converted text into a single string
    str_2 = ""

    for i, chunk in enumerate(chunks):
          
        file_path = os.path.normpath(os.path.join(temp_dir, f"chunk_{i}.wav"))
        
        text = speech_to_text(file_path)
        
        print(text)
        
        if text:  # Only append text if it was successfully recognized
            str_2 += text
        
    # Write the combined text to a file
    text_file_path = 'output.txt'
    with open(text_file_path, 'w') as f:
    
        f.write(str_2)

    # Return the text file as a response
    return send_file(text_file_path, as_attachment=True, download_name='output.txt')


if __name__ == '__main__':
    app.run(debug=True)

    #hi