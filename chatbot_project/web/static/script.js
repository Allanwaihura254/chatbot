// script.js
// Handles the chat UI, talks to the Flask /api/chat endpoint, and
// switches the interface language (English / Kiswahili / French).
// Note: the underlying ML model is trained on English utterances, so
// bot replies stay in English; the UI chrome and greeting switch languages,
// and the rule-based fallback recognises a couple of Swahili keywords
// ("sasa", "asante") already built into rule_engine.py.

const UI_STRINGS = {
  en: {
    title: "TVET Student Support Chatbot",
    langLabel: "Language:",
    placeholder: "Type your message...",
    send: "Send",
    reset: "Reset",
    greeting: "Hello! I'm your student support assistant. How can I help you today?",
    resetNotice: "Conversation reset.",
  },
  sw: {
    title: "Roboti ya Msaada kwa Wanafunzi",
    langLabel: "Lugha:",
    placeholder: "Andika ujumbe wako...",
    send: "Tuma",
    reset: "Anza Upya",
    greeting: "Habari! Mimi ni msaidizi wako wa masuala ya wanafunzi. Nikusaidiaje leo?",
    resetNotice: "Mazungumzo yameanzishwa upya.",
  },
  fr: {
    title: "Chatbot d'Assistance aux Étudiants",
    langLabel: "Langue:",
    placeholder: "Tapez votre message...",
    send: "Envoyer",
    reset: "Réinitialiser",
    greeting: "Bonjour ! Je suis votre assistant d'aide aux étudiants. Comment puis-je vous aider ?",
    resetNotice: "Conversation réinitialisée.",
  },
};

let currentLang = "en";
const sessionId = "session-" + Math.random().toString(36).slice(2);

const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const resetBtn = document.getElementById("reset-btn");
const langSelect = document.getElementById("lang-select");
const titleEl = document.getElementById("title");
const langLabelEl = document.getElementById("lang-label");
const modelInfoEl = document.getElementById("model-info");

function applyLanguage(lang) {
  currentLang = lang;
  const s = UI_STRINGS[lang];
  titleEl.textContent = s.title;
  langLabelEl.textContent = s.langLabel;
  userInput.placeholder = s.placeholder;
  document.getElementById("send-btn").textContent = s.send;
  resetBtn.textContent = s.reset;
}

function addMessage(text, sender, meta) {
  const div = document.createElement("div");
  div.className = `msg ${sender}`;
  div.textContent = text;
  if (meta) {
    const metaSpan = document.createElement("span");
    metaSpan.className = "meta";
    metaSpan.textContent = meta;
    div.appendChild(metaSpan);
  }
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

async function sendMessage(message) {
  addMessage(message, "user");
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId }),
    });
    const data = await res.json();
    if (data.error) {
      addMessage("Error: " + data.error, "bot");
      return;
    }
    addMessage(
      data.response,
      "bot",
      `intent: ${data.intent} | confidence: ${data.confidence}`
    );
  } catch (err) {
    addMessage("Could not reach the server. Is app.py running?", "bot");
  }
}

async function loadModelInfo() {
  try {
    const res = await fetch("/api/model-info");
    const data = await res.json();
    modelInfoEl.textContent =
      `Model: ${data.best_model} | NB acc: ${data.naive_bayes_accuracy} | ` +
      `KNN acc: ${data.knn_accuracy} | Intents: ${data.num_intents}`;
  } catch (err) {
    modelInfoEl.textContent = "";
  }
}

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = userInput.value.trim();
  if (!text) return;
  sendMessage(text);
  userInput.value = "";
});

resetBtn.addEventListener("click", async () => {
  await fetch("/api/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  chatWindow.innerHTML = "";
  addMessage(UI_STRINGS[currentLang].resetNotice, "bot");
  addMessage(UI_STRINGS[currentLang].greeting, "bot");
});

langSelect.addEventListener("change", (e) => {
  applyLanguage(e.target.value);
});

// Initial load
applyLanguage("en");
loadModelInfo();
addMessage(UI_STRINGS.en.greeting, "bot");
