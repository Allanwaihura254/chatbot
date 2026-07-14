"""
chatbot_engine.py
ChatbotEngine: ties preprocessing, classification, the rule-based fallback,
and conversation context tracking into one usable chatbot.
"""

import random
from rule_engine import RuleBasedFallback


class ChatbotEngine:
    """Runs the full chat pipeline for a single conversation session."""

    CONFIDENCE_THRESHOLD = 0.45

    def __init__(self, classifier, responses_map):
        self.classifier = classifier
        self.responses_map = responses_map
        self.rule_fallback = RuleBasedFallback()
        self.history = []  # list of {"user": ..., "intent": ..., "confidence": ...}

    def _resolve_follow_up(self, user_text, predicted_tag, confidence):
        """Simple context resolution: if this looks like a short follow-up
        question and the last turn was about hours, route it to the
        weekend-hours intent instead of misclassifying it standalone."""
        follow_up_markers = ["what about", "and", "how about"]
        text_lower = user_text.lower()

        if self.history and any(m in text_lower for m in follow_up_markers):
            last_intent = self.history[-1]["intent"]
            if last_intent == "hours_query" and (
                "weekend" in text_lower or "saturday" in text_lower or "sunday" in text_lower
            ):
                return "hours_weekend", max(confidence, 0.9)

        return predicted_tag, confidence

    def get_response(self, user_text):
        """Main entry point: classify the message, resolve context,
        apply the confidence threshold, and return (response, confidence)."""
        predicted_tag, confidence = self.classifier.predict(user_text)
        predicted_tag, confidence = self._resolve_follow_up(
            user_text, predicted_tag, confidence
        )

        if confidence >= self.CONFIDENCE_THRESHOLD:
            response = random.choice(self.responses_map.get(predicted_tag, [
                "I'm not sure how to respond to that."
            ]))
        else:
            # Multiple inferencing: fall back to the rule-based module
            response = self.rule_fallback.get_response(user_text)
            predicted_tag = "rule_based_fallback"

        self.history.append({
            "user": user_text,
            "intent": predicted_tag,
            "confidence": confidence,
        })

        return response, confidence

    def reset(self):
        """Clears the conversation history for a new session."""
        self.history = []
