"""
preprocessor.py
TextPreprocessor: tokenisation, lowercasing, punctuation removal,
stop word filtering, and lemmatisation using NLTK.
"""

import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet"):
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg, quiet=True)


class TextPreprocessor:
    """Cleans raw text so it is ready for vectorisation."""

    def __init__(self, language="english"):
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        # Lambda helper for quick punctuation stripping, as required by the brief
        self.strip_punctuation = lambda text: text.translate(
            str.maketrans("", "", string.punctuation)
        )

    def clean(self, text):
        """Runs the full pipeline: lowercase -> strip punctuation ->
        tokenise -> remove stop words -> lemmatise. Returns a cleaned string."""
        text = text.lower()
        text = self.strip_punctuation(text)
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words and t.strip()]
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(tokens)

    def clean_batch(self, texts):
        """Applies clean() to a list of texts."""
        return [self.clean(t) for t in texts]


if __name__ == "__main__":
    pre = TextPreprocessor()
    sample = "What time do offices open on Saturdays??"
    print("Original:", sample)
    print("Cleaned :", pre.clean(sample))
