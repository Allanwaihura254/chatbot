"""
rule_engine.py
RuleBasedFallback: keyword-matching rules used when the ML classifier's
confidence is below the threshold. Each rule below corresponds to a
formal propositional logic expression (write these into your report):

Let:
  G  = message contains a greeting keyword ("hi", "hello", "hey")
  F  = message contains a farewell keyword ("bye", "goodbye")
  T  = message contains a thanks keyword ("thanks", "thank you")
  W  = message contains "weekend"/"saturday"/"sunday"
  H  = message contains "hour"/"time"/"open"/"close"
  FE = message contains "fee"/"pay"/"tuition"
  M  = message contains "mpesa"/"bank"/"pay"
  L  = message contains "library"
  HO = message contains "hostel"/"accommodation"/"room"
  E  = message contains "exam"/"timetable"
  R  = message contains "result"/"grade"/"marks"
  C  = message contains "complain"/"problem"/"issue"

Rule 1: G AND NOT F                        -> greet the user
Rule 2: F AND NOT G                        -> say goodbye
Rule 3: T                                  -> acknowledge thanks
Rule 4: H AND W                            -> weekend hours reply
Rule 5: H AND NOT W                        -> weekday hours reply
Rule 6: FE AND M                           -> fee payment methods reply
Rule 7: FE AND NOT M                       -> general fee query reply
Rule 8: L                                  -> library hours reply
Rule 9: HO                                 -> hostel info reply
Rule 10: E                                 -> exam timetable reply
Rule 11: R                                 -> results query reply
Rule 12: C                                 -> complaint procedure reply
Rule 13: NOT (G OR F OR T OR H OR FE OR L OR HO OR E OR R OR C) -> fallback
"""


class RuleBasedFallback:
    """A small keyword-matching rule engine used when ML confidence is low."""

    def __init__(self):
        self.keywords = {
            "greeting": ["hi", "hello", "hey", "sasa"],
            "farewell": ["bye", "goodbye"],
            "thanks": ["thanks", "thank you", "asante"],
            "weekend": ["weekend", "saturday", "sunday"],
            "hours": ["hour", "time", "open", "close"],
            "fees": ["fee", "tuition", "pay"],
            "payment_method": ["mpesa", "bank", "paybill"],
            "library": ["library"],
            "hostel": ["hostel", "accommodation", "room"],
            "exam": ["exam", "timetable"],
            "results": ["result", "grade", "marks"],
            "complaint": ["complain", "complaint", "problem", "issue"],
        }
        self.default_response = (
            "I'm not fully sure I understood that. Could you rephrase, "
            "or ask about fees, exams, hostels, results or admissions?"
        )

    def _contains_any(self, text, keys):
        return any(k in text for k in keys)

    def get_response(self, text):
        """Applies the rules above in order and returns a response string."""
        text = text.lower()
        g = self._contains_any(text, self.keywords["greeting"])
        f = self._contains_any(text, self.keywords["farewell"])
        t = self._contains_any(text, self.keywords["thanks"])
        w = self._contains_any(text, self.keywords["weekend"])
        h = self._contains_any(text, self.keywords["hours"])
        fe = self._contains_any(text, self.keywords["fees"])
        m = self._contains_any(text, self.keywords["payment_method"])
        lib = self._contains_any(text, self.keywords["library"])
        ho = self._contains_any(text, self.keywords["hostel"])
        e = self._contains_any(text, self.keywords["exam"])
        r = self._contains_any(text, self.keywords["results"])
        c = self._contains_any(text, self.keywords["complaint"])

        # Rule 1
        if g and not f:
            return "Hello! How can I help you today?"
        # Rule 2
        if f and not g:
            return "Goodbye! Take care."
        # Rule 3
        if t:
            return "You're welcome!"
        # Rule 4
        if h and w:
            return "We're open Saturdays 9 AM–1 PM, closed on Sundays."
        # Rule 5
        if h and not w:
            return "Our offices are open Monday to Friday, 8 AM to 5 PM."
        # Rule 6
        if fe and m:
            return "You can pay fees via M-Pesa Paybill or bank deposit."
        # Rule 7
        if fe and not m:
            return "Check the current fee structure with the finance office."
        # Rule 8
        if lib:
            return "The library is open 8 AM–8 PM weekdays, 9 AM–1 PM Saturday."
        # Rule 9
        if ho:
            return "Hostel accommodation is available; apply via student affairs."
        # Rule 10
        if e:
            return "Exam timetables are posted by the exams office before each exam period."
        # Rule 11
        if r:
            return "Check your results on the student portal a few weeks after exams."
        # Rule 12
        if c:
            return "Please file complaints with the student affairs office."
        # Rule 13 (fallback)
        return self.default_response
