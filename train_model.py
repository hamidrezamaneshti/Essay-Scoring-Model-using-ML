import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pickle
import gensim
import spacy

# Read Dataset
df = pd.read_csv('./dataset/Training_Set_rel3.tsv', sep='\t', encoding='ISO-8859-1')
df = df.drop(
    ['rater1_domain1', 'rater2_domain1', 'rater3_domain1', 'domain2_score', 'rater1_domain2', 'rater2_domain2'],
    axis=1)
df = df.fillna(df.mean())
df['normalized_score'] = (df['domain1_score'] - df['domain1_score'].min()) / (
        df['domain1_score'].max() - df['domain1_score'].min())

# Clean the text
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if not word in stop_words]
    text = ' '.join(text)
    return text


df['cleaned_text'] = df['essay'].apply(lambda x: clean_text(x))

# Tokenize the Text
nltk.download('punkt')
df['tokenized_text'] = df['cleaned_text'].apply(lambda x: word_tokenize(x))

# Lemmatize the Text
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()


def lemmatize_text(text):
    text = [lemmatizer.lemmatize(word) for word in text]
    text = ' '.join(text)
    return text


df['lemmatized_text'] = df['tokenized_text'].apply(lambda x: lemmatize_text(x))

# Use word embeddings for feature extraction
tokenized_text = df['tokenized_text'].tolist()
word2vec_model = gensim.models.Word2Vec(tokenized_text, vector_size=100, window=5, min_count=1, workers=4)


def text_to_embeddings(tokenized_text, word2vec_model):
    embeddings = [word2vec_model.wv.get_vector(word) for word in tokenized_text if
                  word in word2vec_model.wv.key_to_index]
    return np.mean(embeddings, axis=0)


df['embeddings'] = df['tokenized_text'].apply(lambda x: text_to_embeddings(x, word2vec_model))
X = np.array(df['embeddings'].tolist())

# Add additional features such as grammar and punctuation
nlp = spacy.load('en_core_web_sm')


def extract_grammar_punctuation_features(text):
    doc = nlp(text)
    num_nouns = len([token for token in doc if token.pos_ == 'NOUN'])
    num_verbs = len([token for token in doc if token.pos_ == 'VERB'])
    num_adjectives = len([token for token in doc if token.pos_ == 'ADJ'])
    num_adverbs = len([token for token in doc if token.pos_ == 'ADV'])
    num_punctuations = len([token for token in doc if token.pos_ == 'PUNCT'])

    return [num_nouns, num_verbs, num_adjectives, num_adverbs, num_punctuations]


df['grammar_punctuation_features'] = df['essay'].apply(lambda x: extract_grammar_punctuation_features(x))
additional_features = df['grammar_punctuation_features'].tolist()
X = np.hstack((X, additional_features))

# Split Data into Training and Test Sets
y = df['normalized_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))

# Compile the Model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

# Train the Model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Calculate min and max scores
min_score = df['domain1_score'].min()
max_score = df['domain1_score'].max()

print("Maximum score in the original dataset:", max_score)

# Save the Word2Vec model
word2vec_model.save("word2vec_model.bin")

# Save the model
model.save('essay_scoring_model.h5')

# Save min and max scores
with open('min_max_scores.pkl', 'wb') as f:
    pickle.dump((min_score, max_score), f)

# Evaluate the Model
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test MSE:', score[1])
