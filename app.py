from flask import Flask, request, render_template
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pickle

# Download stopwords if not already downloaded
nltk.download('stopwords', quiet=True)

# Load the model and vectorizer
clf = pickle.load(open('Sentiment_Analysis_dataset/clf.pkl', 'rb'))
tfidf = pickle.load(open('Sentiment_Analysis_dataset/tdidf.pkl', 'rb'))

app = Flask(__name__)

stopwords_set = set(stopwords.words('english'))  # defining the stopword in the English language
emoji_pattern = re.compile('(?::|;|=)(?:-)?(?:\)|\(|D|P)')  # defining the pattern for the emojis

def preprocessing(text):
    text = re.sub('<[^>]*>', '', text)
    emojis = emoji_pattern.findall(text)
    text = re.sub('[\W+]', ' ', text.lower()) + ' '.join(emojis).replace('-', '')
    porter = PorterStemmer()
    text = [porter.stem(word) for word in text.split() if word not in stopwords_set]
    return " ".join(text)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    prediction = None
    if request.method == 'POST':
        comment = request.form['inputText']
        cleaned_comment = preprocessing(comment)  # Use the cleaned comment
        comment_vector = tfidf.transform([cleaned_comment])
        prediction = clf.predict(comment_vector)[0]

        return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
