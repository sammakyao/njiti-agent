// Chat page: sends questions to /api/chat and streams the answer in.
// Status goes "Thinking..." -> "Reviewing sources..." -> the answer, with its sources listed underneath.

const thread = document.getElementById("thread");
const threadId = newThreadId();
let controller = null;

marked.setOptions({ gfm: true, breaks: true });

// Links in answers open in a new tab
DOMPurify.addHook("afterSanitizeAttributes", (node) => {
  if (node.tagName === "A" && /^https?:/i.test(node.getAttribute("href") || "")) {
    node.setAttribute("target", "_blank");
    node.setAttribute("rel", "noopener noreferrer");
  }
});

const composer = setupComposer(document.getElementById("chat-form"), ask);
composer.sendButton.addEventListener("click", () => {
  if (composer.sendButton.classList.contains("busy") && controller) controller.abort();
});

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

function sourceIcon(source) {
  const icon = element("span", "source-icon");
  if (source.kind === "who") {
    icon.classList.add("who");
    icon.textContent = "WHO";
  } else {
    icon.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6"/></svg>';
  }
  return icon;
}

function sourceChip(source) {
  const url = /^https?:/i.test(source.url || "") ? source.url : null;
  const chip = element(url ? "a" : "span", "source-chip");
  if (url) {
    chip.href = url;
    chip.target = "_blank";
    chip.rel = "noopener noreferrer";
  }
  chip.title = source.title;
  chip.append(sourceIcon(source), element("span", "", source.title));
  return chip;
}

function nearBottom() {
  return window.innerHeight + window.scrollY >= document.body.scrollHeight - 160;
}

function scrollDown() {
  window.scrollTo({ top: document.body.scrollHeight });
}

function createTurn(question) {
  const turn = element("section", "turn");
  const answer = element("div", "answer");
  const status = element("div", "status");
  const icons = element("span", "source-icons");
  const label = element("span", "shimmer", "Thinking...");
  const body = element("div", "markdown");
  const sources = element("div", "answer-sources");
  sources.hidden = true;

  status.append(icons, label);
  answer.append(status, body, sources);
  turn.append(element("div", "user-bubble", question), answer);
  thread.append(turn);

  const seen = new Set();
  let text = "";
  let renderQueued = false;

  const render = () => {
    renderQueued = false;
    const follow = nearBottom();
    body.innerHTML = DOMPurify.sanitize(marked.parse(text));
    if (follow) scrollDown();
  };

  return {
    setStatus(message) {
      status.hidden = false;
      label.textContent = message;
    },
    addSources(list) {
      for (const source of list) {
        const key = source.title + "|" + (source.url || "");
        if (seen.has(key)) continue;
        seen.add(key);
        if (icons.children.length < 3) icons.append(sourceIcon(source));
        sources.append(sourceChip(source));
      }
    },
    // Text the model wrote before deciding to look something up is a preamble, not the answer
    resetText() {
      text = "";
      render();
    },
    addText(piece) {
      status.hidden = true;
      text += piece;
      if (!renderQueued) {
        renderQueued = true;
        requestAnimationFrame(render);
      }
    },
    finish(note) {
      status.hidden = true;
      render();
      if (!text && !note) note = "No answer came back. Please try asking again.";
      if (note) answer.append(element("p", "error-note", note));
      sources.hidden = !text || seen.size === 0;
    },
  };
}

async function ask(question) {
  composer.clear();
  composer.setBusy(true);
  const turn = createTurn(question);
  scrollDown();

  controller = new AbortController();
  let note = null;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: question, thread_id: threadId }),
      signal: controller.signal,
    });
    if (!response.ok || !response.body) throw new Error("HTTP " + response.status);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let split;
      while ((split = buffer.indexOf("\n\n")) !== -1) {
        const raw = buffer.slice(0, split);
        buffer = buffer.slice(split + 2);
        if (!raw.startsWith("data: ")) continue;

        const event = JSON.parse(raw.slice(6));
        if (event.type === "status") turn.setStatus("Thinking...");
        else if (event.type === "searching") {
          turn.resetText();
          turn.setStatus("Reviewing sources...");
        } else if (event.type === "sources") turn.addSources(event.sources);
        else if (event.type === "token") turn.addText(event.text);
        else if (event.type === "error") note = event.message;
      }
    }
  } catch (error) {
    note = error.name === "AbortError" ? "Stopped." : "Couldn't reach Njiti. Check your connection and try again.";
  } finally {
    controller = null;
    turn.finish(note);
    composer.setBusy(false);
    composer.input.focus();
  }
}

// A question typed on the landing page arrives as /chat?q=...
const initial = new URLSearchParams(location.search).get("q");
if (initial && initial.trim()) {
  history.replaceState(null, "", "/chat");
  ask(initial.trim());
}
