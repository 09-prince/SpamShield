import pickle
from flask import Flask, render_template, request
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize the stemmer and stop words set
stemmer = PorterStemmer()
stopwords_set = set(stopwords.words("english"))

# Load the vectorizer and classifier
with open('vect.pkl', 'rb') as file:
    vec = pickle.load(file)

with open('clf.pkl', 'rb') as file:
    clf = pickle.load(file)

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/spam', methods=['POST'])
def find():
    try:
        user_input = request.form.get('user_input')
        if not user_input:
            return render_template('result.html', result="No input provided.", prediction="")

        processed_text = preprocess_text(user_input)
        vectorized_text = vec.transform([processed_text])
        prediction = clf.predict(vectorized_text)

        result = "Spam" if prediction[0] == 1 else "Ham"
        return render_template('result.html', result=result, prediction=user_input)
    except Exception as e:
        return render_template('result.html', result=f"An error occurred: {str(e)}", prediction="")


def preprocess_text(data):
    text = data.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    text = ' '.join(text)
    return text


if __name__ == '__main__':
    app.run(debug=True)
