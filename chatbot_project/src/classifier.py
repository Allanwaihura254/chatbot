"""
classifier.py
IntentClassifier: vectorises training utterances (Bag-of-Words and TF-IDF),
trains Naive Bayes (primary) and KNN (secondary), evaluates both, and
predicts the intent + confidence for new user input.
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


class IntentClassifier:
    """Trains and compares Naive Bayes and KNN intent classifiers."""

    def __init__(self, preprocessor, k_neighbors=3):
        self.preprocessor = preprocessor
        self.count_vectorizer = CountVectorizer()
        self.tfidf_vectorizer = TfidfVectorizer()
        self.nb_model = MultinomialNB()
        self.knn_model = KNeighborsClassifier(n_neighbors=k_neighbors)
        self.best_model_name = None
        self.best_model = None
        self.best_vectorizer = None
        self.results = {}

    def train(self, texts, labels, test_size=0.2, random_state=42):
        """Cleans text, vectorises with both BoW and TF-IDF, trains both
        classifiers on the BoW features (as required), evaluates, and
        selects the best performer."""
        cleaned = self.preprocessor.clean_batch(texts)

        # Bag-of-Words features (used for training the two classifiers)
        X_bow = self.count_vectorizer.fit_transform(cleaned)
        # TF-IDF features (built to demonstrate the difference, per the brief)
        X_tfidf = self.tfidf_vectorizer.fit_transform(cleaned)

        X_train, X_test, y_train, y_test = train_test_split(
            X_bow, labels, test_size=test_size, random_state=random_state, stratify=labels
        )

        self.nb_model.fit(X_train, y_train)
        self.knn_model.fit(X_train, y_train)

        nb_pred = self.nb_model.predict(X_test)
        knn_pred = self.knn_model.predict(X_test)

        self.results = {
            "naive_bayes": {
                "accuracy": accuracy_score(y_test, nb_pred),
                "f1": f1_score(y_test, nb_pred, average="weighted", zero_division=0),
                "confusion_matrix": confusion_matrix(y_test, nb_pred, labels=sorted(set(labels))),
                "y_test": y_test,
                "y_pred": nb_pred,
            },
            "knn": {
                "accuracy": accuracy_score(y_test, knn_pred),
                "f1": f1_score(y_test, knn_pred, average="weighted", zero_division=0),
                "confusion_matrix": confusion_matrix(y_test, knn_pred, labels=sorted(set(labels))),
                "y_test": y_test,
                "y_pred": knn_pred,
            },
        }

        # Bag-of-Words vs TF-IDF demonstration (printed, not used for the
        # final model, as the brief asks us to "demonstrate both approaches")
        self._bow_vs_tfidf_demo(cleaned)

        # Select the best model by weighted F1-score
        if self.results["naive_bayes"]["f1"] >= self.results["knn"]["f1"]:
            self.best_model_name = "naive_bayes"
            self.best_model = self.nb_model
        else:
            self.best_model_name = "knn"
            self.best_model = self.knn_model
        self.best_vectorizer = self.count_vectorizer

        return self.results

    def _bow_vs_tfidf_demo(self, cleaned_texts, sample_index=0):
        """Prints a short comparison of BoW counts vs TF-IDF weights for one
        sample sentence, to satisfy the 'demonstrate both approaches' task."""
        bow_vec = self.count_vectorizer.transform([cleaned_texts[sample_index]])
        tfidf_vec = self.tfidf_vectorizer.transform([cleaned_texts[sample_index]])
        print("\n--- Bag-of-Words vs TF-IDF demo ---")
        print("Sample text:", cleaned_texts[sample_index])
        print("BoW non-zero counts   :", bow_vec.nnz)
        print("TF-IDF non-zero weights:", tfidf_vec.nnz)
        print("(BoW gives raw word counts; TF-IDF down-weights common words")
        print(" across the dataset and up-weights words distinctive to this")
        print(" sentence, so frequent-but-uninformative words matter less.)")

    def predict(self, text):
        """Cleans and classifies new input using the best model. Returns
        (predicted_tag, confidence)."""
        cleaned = self.preprocessor.clean(text)
        vec = self.best_vectorizer.transform([cleaned])

        if self.best_model_name == "naive_bayes":
            probs = self.best_model.predict_proba(vec)[0]
            classes = self.best_model.classes_
            best_idx = probs.argmax()
            return classes[best_idx], float(probs[best_idx])
        else:
            probs = self.best_model.predict_proba(vec)[0]
            classes = self.best_model.classes_
            best_idx = probs.argmax()
            return classes[best_idx], float(probs[best_idx])

    def summary(self):
        """Returns a short text summary comparing the two models."""
        nb = self.results["naive_bayes"]
        knn = self.results["knn"]
        lines = [
            f"Naive Bayes -> accuracy: {nb['accuracy']:.2f}, F1: {nb['f1']:.2f}",
            f"KNN         -> accuracy: {knn['accuracy']:.2f}, F1: {knn['f1']:.2f}",
            f"Selected model: {self.best_model_name} "
            f"(higher weighted F1-score on the held-out test set)",
        ]
        return "\n".join(lines)
