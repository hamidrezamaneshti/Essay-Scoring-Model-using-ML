# Essay-Scoring-Model-using-ML
The essay scoring model in this repository uses a combination of natural language processing techniques and neural networks. The natural language processing techniques involve preprocessing the text, including cleaning, tokenization, and lemmatization. Additionally, word embeddings (Word2Vec) are used to represent words as dense vectors.
The neural network component is responsible for training a regression model that takes the extracted features, including the word embeddings and additional grammar and punctuation features, as input. The neural network architecture consists of multiple layers with dropout regularization.
## Requirements

- Python 3.x
- pandas
- numpy
- nltk
- scikit-learn
- keras
- gensim
- spacy

Install the required dependencies using pip:
`pip install pandas numpy nltk scikit-learn keras gensim spacy`

Download the necessary NLTK and spaCy resources:

`python -m nltk.downloader stopwords punkt wordnet`
`python -m spacy download en_core_web_sm`


## Usage

1. Clone the repository:

`git clone https://github.com/hamidrezamaneshti/essay-scoring-model.git`
`cd essay-scoring-model`


2. Download the essay dataset (Training_Set_rel3.tsv) and place it in the `dataset` directory.

3. Run the script:

`python essay_scoring_model.py`

4. The script will preprocess the dataset, train the essay scoring model, and save the trained models and necessary information for future use.

5. Once the script finishes running, you can use the trained models to predict scores for new essays.

## File Description

- `essay_scoring_model.py`: The main script file that performs data preprocessing, feature extraction, model training, and evaluation.
- `dataset/Training_Set_rel3.tsv`: The essay dataset used for training the model.
- `word2vec_model.bin`: The saved Word2Vec model for word embeddings.
- `essay_scoring_model.h5`: The saved Keras model for essay scoring.
- `min_max_scores.pkl`: A pickle file containing the minimum and maximum scores from the original dataset.

## References

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NLTK Documentation](https://www.nltk.org/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Keras Documentation](https://keras.io/)
- [gensim Documentation](https://radimrehurek.com/gensim/)
- [spaCy Documentation](https://spacy.io/)

## License

This project is licensed under the MIT License.






