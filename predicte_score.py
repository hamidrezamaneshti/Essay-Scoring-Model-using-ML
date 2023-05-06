import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import gensim
import spacy
from keras.models import load_model

# Load model and necessaries files
model = load_model('essay_scoring_model.h5')
with open('min_max_scores.pkl', 'rb') as f:
    min_score, max_score = pickle.load(f)
word2vec_model = gensim.models.Word2Vec.load('word2vec_model.bin')

# Cleaning Text
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
nlp = spacy.load('en_core_web_sm')


def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if not word in stop_words]
    text = ' '.join(text)
    return text


def lemmatize_text(text):
    text = [lemmatizer.lemmatize(word) for word in text]
    text = ' '.join(text)
    return text

# Extract grammar features
def extract_grammar_punctuation_features(text):
    doc = nlp(text)
    num_nouns = len([token for token in doc if token.pos_ == 'NOUN'])
    num_verbs = len([token for token in doc if token.pos_ == 'VERB'])
    num_adjectives = len([token for token in doc if token.pos_ == 'ADJ'])
    num_adverbs = len([token for token in doc if token.pos_ == 'ADV'])
    num_punctuations = len([token for token in doc if token.pos_ == 'PUNCT'])
    return [num_nouns, num_verbs, num_adjectives, num_adverbs, num_punctuations]

# Feeding Text to embedding
def text_to_embeddings(tokenized_text, word2vec_model):
    embeddings = [word2vec_model.wv.get_vector(word) for word in tokenized_text if
                  word in word2vec_model.wv.key_to_index]
    return np.mean(embeddings, axis=0)


def predict_essay_score(essay):
    cleaned_essay = clean_text(essay)
    tokenized_essay = word_tokenize(cleaned_essay)
    lemmatized_essay = lemmatize_text(tokenized_essay)
    embeddings = text_to_embeddings(tokenized_essay, word2vec_model)
    grammar_punctuation_features = extract_grammar_punctuation_features(essay)
    additional_features = np.array(grammar_punctuation_features)
    X = np.hstack((embeddings, additional_features))
    X = X.reshape(1, -1)  # Reshape to match the model input shape
    normalized_score = model.predict(X)[0][0]
    predicted_score = normalized_score * (max_score - min_score) + min_score
    return predicted_score


essay = "sda sd w sd yu sdwe."
predicted_score = predict_essay_score(essay)
print("Predicted Score:", predicted_score)
