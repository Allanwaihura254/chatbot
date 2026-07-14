"""
data_loader.py
IntentDataLoader: loads and validates the intents.json dataset.
"""

import json
import os


class IntentDataLoader:
    """Loads the intents dataset and exposes it in a form ready for
    preprocessing and vectorisation."""

    MIN_INTENTS = 20
    MIN_UTTERANCES_PER_INTENT = 10
    MIN_RESPONSES_PER_INTENT = 5

    def __init__(self, filepath):
        self.filepath = filepath
        self.raw_data = None

    def load(self):
        """Reads the JSON file from disk into self.raw_data."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Intents file not found: {self.filepath}")

        with open(self.filepath, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

        self._validate()
        return self.raw_data

    def _validate(self):
        """Checks the dataset meets the assignment's minimum requirements."""
        intents = self.raw_data.get("intents", [])
        if len(intents) < self.MIN_INTENTS:
            raise ValueError(
                f"Expected at least {self.MIN_INTENTS} intents, found {len(intents)}"
            )
        for intent in intents:
            if len(intent["patterns"]) < self.MIN_UTTERANCES_PER_INTENT:
                raise ValueError(
                    f"Intent '{intent['tag']}' has fewer than "
                    f"{self.MIN_UTTERANCES_PER_INTENT} training utterances"
                )
            if len(intent["responses"]) < self.MIN_RESPONSES_PER_INTENT:
                raise ValueError(
                    f"Intent '{intent['tag']}' has fewer than "
                    f"{self.MIN_RESPONSES_PER_INTENT} response templates"
                )

    def get_training_pairs(self):
        """Returns (texts, labels) flattened across all intents, ready for
        vectorisation and classifier training."""
        texts, labels = [], []
        for intent in self.raw_data["intents"]:
            for utterance in intent["patterns"]:
                texts.append(utterance)
                labels.append(intent["tag"])
        return texts, labels

    def get_responses_map(self):
        """Returns {tag: [responses]} for looking up replies at chat time."""
        return {intent["tag"]: intent["responses"] for intent in self.raw_data["intents"]}

    def get_tags(self):
        """Returns the list of all intent tags."""
        return [intent["tag"] for intent in self.raw_data["intents"]]


if __name__ == "__main__":
    loader = IntentDataLoader("../data/intents.json")
    data = loader.load()
    texts, labels = loader.get_training_pairs()
    print(f"Loaded {len(data['intents'])} intents, {len(texts)} training utterances.")
    print("Sample:", texts[0], "->", labels[0])
