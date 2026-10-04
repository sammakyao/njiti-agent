// Shared behaviour for the landing and chat pages: menu, composer box, voice input.

function setupMenu() {
  const button = document.querySelector(".menu-button");
  const panel = document.querySelector(".menu-panel");
  if (!button || !panel) return;

  const close = () => {
    panel.hidden = true;
    button.setAttribute("aria-expanded", "false");
  };

  button.addEventListener("click", (event) => {
    event.stopPropagation();
    panel.hidden = !panel.hidden;
    button.setAttribute("aria-expanded", String(!panel.hidden));
  });
  panel.addEventListener("click", close);
  document.addEventListener("click", (event) => {
    if (!panel.contains(event.target)) close();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") close();
  });
}

// Wires up a .composer form. onSubmit(text) is called with the trimmed question.
function setupComposer(form, onSubmit) {
  const input = form.querySelector("textarea");
  const send = form.querySelector(".send-button");
  const mic = form.querySelector(".mic-button");

  const resize = () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 200) + "px";
  };

  const update = () => {
    resize();
    if (!send.classList.contains("busy")) send.disabled = !input.value.trim();
  };

  input.addEventListener("input", update);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (send.classList.contains("busy")) return;
    const text = input.value.trim();
    if (!text) return;
    onSubmit(text);
  });

  setupVoice(mic, input, update);
  update();

  return {
    input,
    clear() {
      input.value = "";
      update();
    },
    setBusy(busy) {
      send.classList.toggle("busy", busy);
      send.setAttribute("aria-label", busy ? "Stop answering" : "Ask");
      send.type = busy ? "button" : "submit";
      update();
      if (busy) send.disabled = false;
    },
    sendButton: send,
  };
}

// Voice input with the browser's speech recognition. The mic button stays hidden where it isn't supported.
function setupVoice(button, input, onChange) {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!button || !Recognition) return;

  button.hidden = false;
  let recognition = null;

  button.addEventListener("click", () => {
    if (recognition) {
      recognition.stop();
      return;
    }

    recognition = new Recognition();
    recognition.lang = navigator.language || "en-US";
    recognition.interimResults = true;
    const before = input.value ? input.value.trimEnd() + " " : "";

    recognition.onresult = (event) => {
      const spoken = Array.from(event.results).map((result) => result[0].transcript).join("");
      input.value = before + spoken;
      onChange();
    };
    recognition.onend = () => {
      recognition = null;
      button.classList.remove("listening");
      button.setAttribute("aria-label", "Ask with your voice");
      input.focus();
    };

    button.classList.add("listening");
    button.setAttribute("aria-label", "Stop listening");
    recognition.start();
  });
}

function newThreadId() {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID().replace(/-/g, "");
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 12);
}

document.addEventListener("DOMContentLoaded", setupMenu);
