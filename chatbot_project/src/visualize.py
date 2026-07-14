"""
visualize.py
Produces the 4 required charts:
  1. Intent distribution across training data
  2. Model accuracy comparison (Naive Bayes vs KNN)
  3. Confusion matrix for intent classification
  4. Response confidence score distribution across a 30-message test run
"""

import random
import matplotlib.pyplot as plt
import numpy as np

from data_loader import IntentDataLoader
from preprocessor import TextPreprocessor
from classifier import IntentClassifier
from chatbot_engine import ChatbotEngine

OUT_DIR = "../visualizations"


def chart_intent_distribution(loader):
    data = loader.raw_data["intents"]
    tags = [i["tag"] for i in data]
    counts = [len(i["patterns"]) for i in data]

    plt.figure(figsize=(10, 6))
    plt.barh(tags, counts, color="#2E86AB")
    plt.xlabel("Number of training utterances")
    plt.title("Intent Distribution Across Training Data")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/1_intent_distribution.png", dpi=150)
    plt.close()


def chart_model_comparison(results):
    models = ["Naive Bayes", "KNN"]
    accuracy = [results["naive_bayes"]["accuracy"], results["knn"]["accuracy"]]
    f1 = [results["naive_bayes"]["f1"], results["knn"]["f1"]]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar(x - width / 2, accuracy, width, label="Accuracy", color="#2E86AB")
    plt.bar(x + width / 2, f1, width, label="F1-score", color="#F18F01")
    plt.xticks(x, models)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Model Accuracy Comparison: Naive Bayes vs KNN")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/2_model_comparison.png", dpi=150)
    plt.close()


def chart_confusion_matrix(results, tags):
    cm = results["naive_bayes"]["confusion_matrix"]
    labels_sorted = sorted(set(tags))

    plt.figure(figsize=(10, 9))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(labels_sorted)), labels_sorted, rotation=90)
    plt.yticks(range(len(labels_sorted)), labels_sorted)
    plt.xlabel("Predicted intent")
    plt.ylabel("Actual intent")
    plt.title("Confusion Matrix - Naive Bayes Intent Classification")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/3_confusion_matrix.png", dpi=150)
    plt.close()


def chart_confidence_distribution(engine, loader):
    # Build a 30-message test conversation sampled from the dataset
    all_texts, _ = loader.get_training_pairs()
    random.seed(7)
    test_messages = random.sample(all_texts, 30)

    confidences = []
    for msg in test_messages:
        _, conf = engine.get_response(msg)
        confidences.append(conf)

    plt.figure(figsize=(8, 5))
    plt.hist(confidences, bins=10, color="#A23B72", edgecolor="black")
    plt.xlabel("Confidence score")
    plt.ylabel("Number of messages")
    plt.title("Response Confidence Distribution (30-message test run)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/4_confidence_distribution.png", dpi=150)
    plt.close()


def main():
    loader = IntentDataLoader("../data/intents.json")
    loader.load()
    texts, labels = loader.get_training_pairs()
    responses_map = loader.get_responses_map()

    preprocessor = TextPreprocessor()
    classifier = IntentClassifier(preprocessor, k_neighbors=3)
    results = classifier.train(texts, labels)

    engine = ChatbotEngine(classifier, responses_map)

    chart_intent_distribution(loader)
    chart_model_comparison(results)
    chart_confusion_matrix(results, labels)
    chart_confidence_distribution(engine, loader)

    print("All 4 charts saved to the visualizations/ folder.")


if __name__ == "__main__":
    main()
