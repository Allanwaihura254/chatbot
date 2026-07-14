# TVET Student Support Chatbot — Project 2

AI-powered chatbot for a Kenyan TVET institution's student support desk.
Built for the Diploma-level AI capstone (ICT/CU/CS/CR/08/6/A), Project 2.

## Setup

```bash
pip install -r requirements.txt
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Run the chatbot

```bash
cd src
python3 main.py
```

## Generate the visualization dashboard

```bash
cd src
python3 visualize.py
```

Charts are saved to `visualizations/`.

## Run the browser version (Chrome)

The same Python engine (data_loader, preprocessor, classifier,
chatbot_engine) is served over HTTP by a small Flask app, with an
HTML/CSS/JavaScript front end on top. This does not replace the required
CLI deliverable (`main.py`) — it's an extra demo layer for presentation.

```bash
cd web
python3 app.py
```

Then open **http://127.0.0.1:5000** in Chrome. The UI has a language
switcher (English / Kiswahili / Français) for the interface labels and
greeting; the ML model itself is trained on English utterances, so bot
replies are in English, with a couple of Swahili keywords ("sasa",
"asante") already recognised by the rule-based fallback.

## Project structure

```
chatbot_project/
├── data/
│   ├── build_intents.py   # generates intents.json
│   └── intents.json       # 20 intents, 10 utterances, 5 responses each
├── src/
│   ├── data_loader.py     # IntentDataLoader
│   ├── preprocessor.py    # TextPreprocessor (NLTK pipeline)
│   ├── classifier.py      # IntentClassifier (Naive Bayes + KNN)
│   ├── rule_engine.py     # RuleBasedFallback (8+ propositional rules)
│   ├── chatbot_engine.py  # ChatbotEngine (context + fallback logic)
│   ├── main.py             # CLI chat interface (required deliverable)
│   └── visualize.py        # generates the 4 required charts
├── web/
│   ├── app.py               # Flask server wrapping the same engine
│   ├── templates/index.html # chat UI page
│   └── static/
│       ├── style.css
│       └── script.js        # chat logic + language switcher
├── visualizations/         # output charts (generated)
└── requirements.txt
```

## Still to do for full marks

- Write the 1–2 page essay on chatbot history (ELIZA, IBM Watson, Dialogflow).
- Write up the 8 propositional logic rules formally in the report (already
  documented as comments in `rule_engine.py` — copy/adapt them).
- Write the one-page ethical analysis (bias, privacy, over-reliance risk).
- Compile the final report (intro, background, dataset design, architecture,
  NLP pipeline, ML results, ethics, conclusion) — minimum six pages.
- Prepare the 10–15 minute oral demo.
