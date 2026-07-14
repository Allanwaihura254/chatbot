"""
main.py
Interactive command-line chatbot. Run this file to chat with the bot.
"""

from data_loader import IntentDataLoader
from preprocessor import TextPreprocessor
from classifier import IntentClassifier
from chatbot_engine import ChatbotEngine

INTENTS_PATH = "../data/intents.json"
EXIT_COMMANDS = {"quit", "exit", "bye", "goodbye"}


def build_chatbot():
    loader = IntentDataLoader(INTENTS_PATH)
    data = loader.load()
    texts, labels = loader.get_training_pairs()
    responses_map = loader.get_responses_map()

    preprocessor = TextPreprocessor()
    classifier = IntentClassifier(preprocessor, k_neighbors=3)
    classifier.train(texts, labels)

    print("\n=== Model comparison ===")
    print(classifier.summary())

    return ChatbotEngine(classifier, responses_map)


def run_chat():
    print("Building and training the chatbot, please wait...")
    engine = build_chatbot()

    print("\n=== TVET Student Support Chatbot ===")
    print("Type 'quit' or 'bye' to end the conversation.\n")
    print("Bot: Hello! I'm your student support assistant. How can I help you today?")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in EXIT_COMMANDS:
            print("Bot: Goodbye! All the best with your studies.")
            break

        response, confidence = engine.get_response(user_input)
        print(f"Bot: {response}  (confidence: {confidence:.2f})")


if __name__ == "__main__":
    run_chat()
