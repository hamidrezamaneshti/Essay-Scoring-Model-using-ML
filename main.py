import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
import joblib
import pickle

# Read Dataset
df = pd.read_csv('./dataset/Training_Set_rel3.tsv', sep='\t', encoding='ISO-8859-1')
#  Remove Unnecessary Columns, in this case, we can remove these columns as they are not relevant for our analysis.
df = df.drop(
    ['rater1_domain1', 'rater2_domain1', 'rater3_domain1', 'domain2_score', 'rater1_domain2', 'rater2_domain2'], axis=1)
# Handle Missing Values
df = df.fillna(df.mean())
# Normalize the Scores, We normalize the scores to a scale of 0 to 1
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

# Extract features from the preprocessed dataset
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['lemmatized_text']).toarray()

# I use Term Frequency-Inverse Document Frequency technique represent textual data for machine learning tasks
transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()


# Add additional features such as the length of the essay and the average word length.

def extract_additional_features(text):
    length = len(text.split())
    avg_word_length = sum(len(word) for word in text.split()) / length
    return [length, avg_word_length]


additional_features = df['lemmatized_text'].apply(lambda x: extract_additional_features(x)).tolist()
X = np.hstack((X, additional_features))

# Split Data into Training and Test Sets
y = df['normalized_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
'''This will split the dataset into training and test sets, with 80% of the data used for training and 20% for 
testing. '''

# Define the Model: I define the neural network model using the Sequential class from Keras
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))

# Compile the Model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

# Train the Model: now train the neural network model on the training data using the fit method.
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Calculate min and max scores
min_score = df['domain1_score'].min()
max_score = df['domain1_score'].max()

print("Maximum score in the original dataset:", max_score)
# Save the model
model.save('essay_scoring_model.h5')

# Save the fitted vectorizer and transformer objects
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('transformer.pkl', 'wb') as f:
    pickle.dump(transformer, f)

# Save min and max scores
with open('min_max_scores.pkl', 'wb') as f:
    pickle.dump((min_score, max_score), f)

# Evaluate the Model: evaluate the performance of the model on the testing data using the evaluate method.
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test MSE:', score[1])




