# Njiti

An AI answers site about premature babies ("njiti" is Swahili for a baby born too early).
Ask a question on the home page and the chat page shows the agent thinking, reviewing sources, and then
streams back an answer with links to where it came from.

## Run it

Needs Python 3.10 or newer (the Python that ships with macOS is 3.9: use `brew install python@3.12`).

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your keys

uvicorn src.server:app --reload
```

Open http://localhost:8000.

To see the website without any API keys, run `NJITI_DEMO=1 uvicorn src.server:app --reload`. It streams a
canned answer instead of calling a model.

## How it fits together

| Path | What it does |
| --- | --- |
| `src/server.py` | FastAPI app. Serves the pages and `POST /api/chat`, which streams the agent's progress as Server-Sent Events. |
| `src/premature_ai.py` | The LangGraph agent: system prompt, WHO context, and the two tools (`get_context`, `search_preterm_knowledge`). Run it directly for a console chat. |
| `src/rag/BQ_db.py` | BigQuery vector store: load PDFs from `src/docs/` and search them. |
| `src/main.py` | Model and embedding helpers, settings from the environment. |
| `src/web/` | The website: `index.html` (home), `chat.html` + `static/chat.js` (chat), shared styles and scripts. |

## Knowledge base

Put PDFs (WHO guidelines, clinical protocols) in `src/docs/`, set the `BQ_*` variables, then run:

```bash
python -c "from src.rag.BQ_db import create_knowledge_base; create_knowledge_base()"
```

Search results show up on the site as source chips with the document name and page.

## Notes

- Conversation memory is kept in memory on the server (`InMemorySaver`), so it resets when the server restarts
  and isn't shared between multiple server processes. Use a persistent LangGraph checkpointer for production.
- Njiti gives general information, not medical advice. The prompt tells it to send people to a health
  facility first when they describe danger signs.
