import pickle
import gradio as gr
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from transformers import AutoTokenizer

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

with open('mnb_model.pkl', 'rb') as model_file:
    mnb = pickle.load(model_file)


checkpoint = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tf_idf = TfidfVectorizer()

nlp = spacy.load("en_core_web_sm")

class TextPreprocessing:
    def __init__(self, text: str, tokenizer, tfidf_vectorizer: TfidfVectorizer = None):
        self.text = text
        self.tokenizer = tokenizer
        self.tfidf_vectorizer = tfidf_vectorizer or TfidfVectorizer()

    @staticmethod
    def Cleaning_text(text: str) -> str:
        """
        Cleans the input text by converting to lowercase,
        removing URLs, special characters, and unnecessary spaces.
        """
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r"[^a-zA-Z\s]", '', text)
        text = re.sub(r"n't", ' not', text)
        text = re.sub(r"'s", '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def Tokenization_text(text: str) -> list:
        """
        Tokenizes the text into a list of words, excluding punctuations and spaces.
        """
        doc = nlp(text)
        tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
        return tokens

    @staticmethod
    def Lemmatization_text(text: str) -> str:
        """
        Performs lemmatization on the text and returns the lemmatized version.
        """
        doc = nlp(text)
        lemmatized_text = ' '.join([token.lemma_ for token in doc if not token.is_punct and not token.is_space])
        return lemmatized_text

    @staticmethod
    def Stopwords_removal(text: str) -> str:
        """
        Removes stopwords from the input text.
        """
        doc = nlp(text)
        text_without_stopwords = ' '.join([token.text for token in doc if not token.is_stop])
        return text_without_stopwords

    def ModernBert_Tokenization(self) -> dict:
        """
        Tokenizes the cleaned text using ModernBERT's tokenizer.
        """
        cleaned_text = self.Cleaning_text(self.text)
        tokenized_output = self.tokenizer(cleaned_text, return_tensors='pt', truncation=True, padding=True)
        return tokenized_output

    def Tfidf_Transformation(self, texts: list) -> np.ndarray:
        """
        Applies TF-IDF transformation to a list of texts.

        Args:
            texts (list of str): List of text strings to apply the TF-IDF transformation.

        Returns:
            np.ndarray: TF-IDF feature matrix.
        """
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        return tfidf_matrix.toarray()

    def BagOfWords_Transformation(self, texts: list) -> np.ndarray:
        """
        Applies Bag of Words (BoW) transformation to a list of texts.

        Args:
            texts (list of str): List of text strings to apply the BoW transformation.

        Returns:
            np.ndarray: Bag of Words feature matrix.
        """
        vectorizer = CountVectorizer()
        bow_matrix = vectorizer.fit_transform(texts)
        return bow_matrix.toarray()

    def Ngram_Transformation(self, texts: list, ngram_range=(1, 2)) -> np.ndarray:
        """
        Applies N-gram transformation (uni-grams, bi-grams, etc.) to a list of texts.

        Args:
            texts (list of str): List of text strings to apply the N-gram transformation.
            ngram_range (tuple): The range of n-values for n-grams to extract. Default is (1, 2) for unigrams and bigrams.

        Returns:
            np.ndarray: N-gram feature matrix.
        """
        vectorizer = CountVectorizer(ngram_range=ngram_range)
        ngram_matrix = vectorizer.fit_transform(texts)
        return ngram_matrix.toarray()

    


def preprocess_text(text):
    text_preprocessor = TextPreprocessing(text=None, tokenizer=None)
    cleaned_text = text_preprocessor.Cleaning_text(text)
    return cleaned_text


def predict_news(text):
    cleaned_text = preprocess_text(text)
    X_input = tfidf_vectorizer.transform([cleaned_text])
    prediction = mnb.predict(X_input)
    return "Fake News" if prediction == 0 else "Real News"


iface = gr.Interface(
    fn=predict_news, 
    inputs=gr.Textbox(lines=7, placeholder="Enter the news article here..."), 
    outputs="text", 
    title="Fake News Classification", 
    description="Classify news articles as real or fake."
)

iface.launch()
