# Discord RAG Bot — Project Spec

## Goal
A Discord bot that answers user questions by retrieving relevant chunks from
a local knowledge base and generating a response via LLM.

## Tech Stack
- Bot framework: discord.py (v2.x)
- Embeddings: sentence-transformers (all-MiniLM-L6-v2)
- Vector store: SQLite via sqlite-vec
- LLM: Ollama (local, free) — model: llama3.2 or mistral
- Ollama runs locally on your machine, no API key needed
- Language: Python 3.11+

## Bot Commands
- /ask <query>     → RAG pipeline → reply with answer + source doc name
- /image           → user uploads image → reply with caption + 3 tags (Option B, optional)
- /help            → show usage

## Project Structure
discord-rag-bot/
├── bot/
│   ├── __init__.py
│   ├── commands.py        # Discord slash command handlers
│   └── history.py         # Per-user message history (last 3 turns)
├── rag/
│   ├── __init__.py
│   ├── embedder.py        # Embed chunks using sentence-transformers
│   ├── retriever.py       # Query sqlite-vec, return top-k chunks
│   ├── ingester.py        # Load docs → split → embed → store in DB
│   └── llm.py             # Build prompt + call OpenAI API
├── knowledge_base/
│   ├── company_faq.md
│   ├── product_guide.md
│   └── return_policy.md
├── data/
│   └── vectors.db         # SQLite DB (auto-created)
├── main.py                # Entry point
├── config.py              # Env vars loader
├── requirements.txt
├── .env.example
└── README.md

## Data Flow
User types /ask "What is the return policy?"
  → commands.py receives query
  → retriever.py embeds query → cosine search in sqlite-vec → top 3 chunks
  → llm.py builds prompt: [context chunks] + [user history] + [query]
  → Ollama generates answer
  → reply includes: answer + "Source: return_policy.md"

## Key Rules for Codex
- Each file should do ONE thing only (single responsibility)
- No hardcoded secrets — use .env file
- All DB operations go through retriever.py and ingester.py only
- User history is stored in-memory as a dict: {user_id: [last 3 messages]}
- Caching: cache query embeddings in a simple dict to avoid re-embedding
- Error handling: all Discord commands must have try/except with user-friendly error messages