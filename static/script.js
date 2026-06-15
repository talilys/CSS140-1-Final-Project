/* EaseAI — script.js */

const chatMessages = document.getElementById("chat-messages");
const chatScroll   = document.getElementById("chat-scroll");
const userInput    = document.getElementById("user-input");
const sendBtn      = document.getElementById("send-btn");
const menuToggle   = document.getElementById("menu-toggle");
const sidebar      = document.querySelector(".sidebar");

// Get the user's logged-in initial dynamically from the template profile card
function getUserInitial() {
  const profileAvatar = document.querySelector(".user-profile-section .user-avatar");
  return profileAvatar ? profileAvatar.textContent.trim() : "U";
}

// Sidebar toggle (mobile)
menuToggle.addEventListener("click", () => {
  sidebar.classList.toggle("open");
});

document.addEventListener("click", (e) => {
  if (sidebar.classList.contains("open") &&
      !sidebar.contains(e.target) &&
      !menuToggle.contains(e.target)) {
    sidebar.classList.remove("open");
  }
});

// Clear browser storage on logout to keep data safe between user switches
const logoutBtn = document.querySelector(".logout-btn");
if (logoutBtn) {
  logoutBtn.addEventListener("click", () => {
    localStorage.clear();
    sessionStorage.clear();
  });
}

// Auto-resize textarea
userInput.addEventListener("input", () => {
  userInput.style.height = "auto";
  userInput.style.height = Math.min(userInput.scrollHeight, 120) + "px";
});

// Scroll to bottom
function scrollBottom() {
  chatScroll.scrollTop = chatScroll.scrollHeight;
}

// Load chat history on page load
async function loadChatHistory() {
  try {
    const res = await fetch("/get_history");
    const data = await res.json();
    
    if (data.history && data.history.length > 0) {
      // Clear the initial greeting message
      chatMessages.innerHTML = "";
      
      // Load all historical messages
      data.history.forEach(msg => {
        addUserMessage(msg.user_message);
        addBotMessage(msg.bot_response, msg.sentiment, msg.confidence);
      });
    }
  } catch (err) {
    console.error("Error loading chat history:", err);
  }
}

// Load history when page loads
document.addEventListener("DOMContentLoaded", loadChatHistory);

// Add a user bubble using the dynamic user initial
function addUserMessage(text) {
  const row = document.createElement("div");
  row.className = "msg-row user-row";

  const avatar = document.createElement("div");
  avatar.className = "avatar user-avatar";
  avatar.textContent = getUserInitial(); // Uses dynamic letter instead of hardcoded "U"

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble user-bubble";
  bubble.textContent = text;

  row.appendChild(bubble);
  row.appendChild(avatar);
  chatMessages.appendChild(row);
  scrollBottom();
}

// Add typing indicator
function addTypingIndicator() {
  const row = document.createElement("div");
  row.className = "typing-row";
  row.id = "typing-row";

  const avatar = document.createElement("div");
  avatar.className = "avatar bot-avatar";
  avatar.textContent = "E";

  const bubble = document.createElement("div");
  bubble.className = "typing-bubble";
  bubble.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';

  row.appendChild(avatar);
  row.appendChild(bubble);
  chatMessages.appendChild(row);
  scrollBottom();
}

function removeTypingIndicator() {
  const el = document.getElementById("typing-row");
  if (el) el.remove();
}

// Sentiment label map
const SENTIMENT_LABELS = {
  positive: "Feeling positive",
  negative: "Stressed / low mood",
  neutral:  "Neutral",
  unclear:  "Tell me more",
  crisis:   "Please seek support",
  sarcasm:  "I see you 👀 — how are you really?",
};

// Add a bot bubble
function addBotMessage(text, sentiment, confidence) {
  const row = document.createElement("div");
  row.className = "msg-row bot-row";

  const avatar = document.createElement("div");
  avatar.className = "avatar bot-avatar";
  avatar.textContent = "E";

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble bot-bubble";

  const p = document.createElement("p");
  p.textContent = text;
  bubble.appendChild(p);

  if (sentiment) {
    const chip = document.createElement("div");
    chip.className = "mood-chip chip-" + sentiment;
    chip.textContent = SENTIMENT_LABELS[sentiment] || sentiment;

    if (sentiment !== "unclear" && sentiment !== "crisis" && confidence !== null) {
      const conf = document.createElement("span");
      conf.className = "chip-confidence";
      conf.textContent = " · " + confidence + "%";
      chip.appendChild(conf);
    }

    bubble.appendChild(chip);
  }

  row.appendChild(avatar);
  row.appendChild(bubble);
  chatMessages.appendChild(row);
  scrollBottom();
}

// Send message — guarded with isSending flag
let isSending = false;

async function sendMessage() {
  if (isSending) return;

  const message = userInput.value.trim();
  if (!message) return;

  isSending = true;
  sendBtn.disabled = true;

  addUserMessage(message);
  userInput.value = "";
  userInput.style.height = "auto";

  addTypingIndicator();
  await new Promise(function(r) { setTimeout(r, 600); });

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: message }),
    });

    const data = await res.json();
    removeTypingIndicator();

    if (data.error) {
      addBotMessage("Sorry, something went wrong. Please try again.", null, null);
    } else {
      addBotMessage(data.response, data.sentiment, data.confidence);
    }

  } catch (err) {
    removeTypingIndicator();
    addBotMessage("Oops! I couldn't connect to the server.", null, null);
    console.error(err);
  }

  isSending = false;
  sendBtn.disabled = false;
}

// Clear chat history
async function clearChatHistory() {
  if (!confirm("Are you sure you want to delete all chat history? This cannot be undone.")) {
    return;
  }

  try {
    const res = await fetch("/clear_history", { method: "POST" });
    const data = await res.json();

    if (data.success) {
      chatMessages.innerHTML = "";
      const row = document.createElement("div");
      row.className = "msg-row bot-row";
      const avatar = document.createElement("div");
      avatar.className = "avatar bot-avatar";
      avatar.textContent = "E";
      const bubble = document.createElement("div");
      bubble.className = "msg-bubble bot-bubble";
      const p = document.createElement("p");
      p.textContent = "Hi there 👋 I'm EaseAI. I'm here to listen and support you through the ups and downs of school life.";
      bubble.appendChild(p);
      row.appendChild(avatar);
      row.appendChild(bubble);
      chatMessages.appendChild(row);
      scrollBottom();
    }
  } catch (err) {
    alert("Error clearing history. Please try again.");
    console.error(err);
  }
}

// Event listeners
sendBtn.addEventListener("click", sendMessage);

userInput.addEventListener("keydown", function(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// Clear history button
const clearHistoryBtn = document.getElementById("clear-history-btn");
if (clearHistoryBtn) {
  clearHistoryBtn.addEventListener("click", clearChatHistory);
}