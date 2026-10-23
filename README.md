# CiteDocs

A small retrieval-augmented generation (RAG) service that answers questions about a set of product-documentation pages. Every answer cites the pages it came from, and when the documentation does not support an answer, the service says **"I don't know based on the provided documentation"** instead of guessing.

It includes an evaluation harness that measures retrieval and citation accuracy, so claims about quality come from numbers the script prints.

## Why this exists

Documentation assistants fail in two ways: they invent answers, and they give answers nobody can check. CiteDocs is built around the opposite rules:

- **Grounded:** the model only sees retrieved documentation chunks and is told to use nothing else.
- **Cited:** every answer lists the pages it relied on, and citations that point to chunks that were not retrieved are rejected.
- **Honest:** if the best match is not similar enough, the service answers "I don't know" **without calling the LLM**.
- **Measured:** `python -m eval.run` reports hit rate, citation accuracy, "I don't know" accuracy and latency on a labelled question set.

## Status

| Area | State |
|---|---|
| Config, ingestion, retrieval | Done |
| LLM layer and API (`/ask`, `/summarize`, `/sources`, `/health`) | Done |
| Offline test suite and linting | Done |
| Evaluation harness | Done; results depend on the docs you ingest (see [Evaluation](#evaluation)) |
| Dockerfile and Docker Compose | Planned |
| CI/CD, CodeQL, Dependabot | Planned |

## Architecture

```mermaid
flowchart LR
  subgraph Ingest["Ingest: python -m app.ingest"]
    D["data/docs/*.md<br/>front matter: source_url, title"] --> P["Parse and chunk<br/>tokens + overlap"]
    P --> E1["Embed title + text<br/>MiniLM, local"]
    E1 --> I[("index/<br/>FAISS + chunks.json + meta.json")]
  end
  subgraph Query["Query path"]
    C["Client"] --> API["FastAPI<br/>validate input, rate limit"]
    API --> R["Retriever<br/>embed question, top-k cosine"]
    I --> R
    R --> T{"best score >= threshold?"}
    T -- no --> N["answered=false<br/>no LLM call"]
    T -- yes --> G["Groq LLM<br/>context-only prompt, JSON"]
    G --> V{"valid JSON and cited IDs were retrieved?"}
    V -- "no: retry once" --> G
    V -- "still no" --> N
    V -- yes --> A["answer + citations + confidence"]
  end
```

**Ingestion** reads local `.md` / `.txt` files only (no scraping), splits each page into overlapping token windows, embeds each chunk locally and writes a FAISS index to `./index/`.

**Querying** embeds the question, finds the top-k chunks by cosine similarity and compares the best score with a threshold. Below it, the service returns `answered=false`. Above it, the chunks go to Groq with a prompt that restricts the answer to that context and requires JSON naming the chunk IDs used. The reply is validated with Pydantic, and its chunk IDs are checked against the chunks that were actually retrieved.

## Requirements

- Python 3.11+
- A free [Groq](https://console.groq.com) API key and a chat model ID from Groq's model list
- About 100 MB of disk for the embedding model, downloaded on first use
- macOS (arm64) or Linux

## Setup (local)

```bash
git clone https://github.com/HajiraS/CiteDocs.git
cd CiteDocs
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

cp .env.example .env
```

Edit `.env` and set `GROQ_API_KEY` and `GROQ_MODEL`. `.env` is git-ignored. Never put real values in `.env.example`.

## Adding documentation

1. Save each page as a `.md` or `.txt` file in `data/docs/`. Every file must start with front matter containing the page's `source_url` and `title`:

   ```markdown
   ---
   source_url: https://docs.example.com/dhcp/failover
   title: "Configuring DHCP Failover"
   ---
   Page text starts here...
   ```

   Quote any title that contains a colon, or the YAML parser will reject it.

2. Build the index:

   ```bash
   python -m app.ingest
   ```

   The command prints chunks per document and the total. Re-run it whenever the files change. The index records which embedding model built it, and the service refuses to load an index built with a different one.

Use public documentation you are allowed to copy. Check the publisher's terms before committing page text to a public repository.

## Running the API

```bash
uvicorn app.main:app --port 8000
```

Use any free port. If the index or the Groq settings are missing, the service still starts: `/health` reports `"status": "degraded"` and the affected endpoints return a clear 503.

### `POST /ask`

```bash
curl -s -X POST localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "How does DHCP failover work?"}'
```

Response shape:

```json
{
  "answer": "<text, or the I-don't-know message>",
  "citations": [{"title": "<page title>", "source_url": "<page url>", "snippet": "<up to ~300 chars>"}],
  "confidence": 0.0,
  "answered": true
}
```

`confidence` is the best retrieval similarity clamped to [0, 1]. It measures how close the question is to the docs. It is not a calibrated probability that the answer is correct.

### `POST /summarize`

```bash
curl -s -X POST localhost:8000/summarize \
  -H 'Content-Type: application/json' \
  -d '{"topic": "DNS firewall"}'
```

Returns `topic`, 3 to 5 `bullets`, `citations`, `confidence` and `answered`. It uses the same retrieval, threshold and grounding rules as `/ask`. When it cannot answer, `bullets` is empty and `message` explains why.

### `GET /sources` and `GET /health`

```bash
curl -s localhost:8000/sources   # indexed documents with chunk counts
curl -s localhost:8000/health    # {"status": "ok" | "degraded", "index_loaded": bool, "chunks": int}
```

### Errors

| Status | When |
|---|---|
| 200 with `answered: false` | Nothing in the docs matches well enough, or the model's output stayed invalid after one retry |
| 422 | Empty question, only whitespace or control characters, or longer than `MAX_QUESTION_LENGTH` |
| 429 | Rate limit exceeded; includes a `Retry-After` header |
| 503 | Index not loaded, Groq not configured, or Groq unavailable after retries |

## Configuration

All settings are environment variables (or entries in `.env`).

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | none | Groq API key (required for answers) |
| `GROQ_MODEL` | none | Groq chat model ID (required for answers) |
| `TOP_K` | 4 | Chunks retrieved per question |
| `SIMILARITY_THRESHOLD` | 0.30 | Minimum best similarity to call the LLM |
| `CHUNK_TOKENS` | 500 | Chunk size, in embedding-model tokens |
| `CHUNK_OVERLAP` | 50 | Tokens shared by neighbouring chunks |
| `MAX_QUESTION_LENGTH` | 500 | Longest accepted question or topic, in characters |
| `RATE_LIMIT_PER_MINUTE` | 20 | Requests per minute per client IP on `/ask` and `/summarize` |
| `LLM_TIMEOUT_SECONDS` | 20 | Per-request timeout for Groq |
| `LLM_MAX_RETRIES` | 3 | Retries with backoff on timeouts, 429s and 5xx errors |
| `DOCS_DIR` | `data/docs` | Where ingestion reads pages from |
| `INDEX_DIR` | `index` | Where the index is written and loaded from |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |

## Tests

```bash
ruff check .
pytest
```

The tests run offline and need no API key. They use a deterministic fake embedder over a tiny corpus and a scripted fake LLM, with a real FAISS index. Coverage is reported by `pytest` (configured in `pytest.ini`). Cover areas: chunking, the threshold fallback, citation validation, invalid-JSON retry, input validation, rate limiting, Groq backoff, and the `/health` and `/sources` endpoints.

Code that needs a model download or a live network call (the real embedding model and the Groq network path) is not covered by unit tests. The evaluation exercises it.

## Evaluation

`eval/questions.jsonl` holds the labelled questions, one JSON object per line: answerable questions carry the `expected_source_url` of the single page that answers them, and unanswerable questions have `null`.

```bash
python -m eval.run --check     # validate the questions against data/docs (no model or key)
python -m eval.run --no-llm    # retrieval only, no Groq key needed
python -m eval.run --delay 2   # full run; --delay spaces out Groq calls for free-tier limits
python -m eval.run --help      # threshold and chunk-size sweeps and other options
```

Results are printed and saved to `eval/results.json` (git-ignored). Metric definitions:

- **Retrieval hit rate:** the expected page appears in the top-k chunks.
- **Citation accuracy:** the question was answered and the expected page is among the citations. This checks that the right page was cited, not that the answer text is factually correct.
- **Correct "I don't know" rate:** an unanswerable question came back with `answered=false`.
- **Average latency:** retrieval plus the LLM call, per question.

If a Groq call fails after retries, that question is excluded from the rates and counted as an LLM failure rather than a wrong answer.

### Results

Add these from `eval/results.json` after running against your own docs, with the date, model, threshold and chunk size you used:

| Metric | Value |
|---|---|
| Retrieval hit rate | |
| Citation accuracy | |
| Correct "I don't know" rate | |
| Average latency | |

With about 20 questions, one question moves an answerable rate by several points, so treat small differences as noise. If you chose the threshold from a sweep on the same questions, say so here.

## Design decisions

**Local embeddings.** Groq offers no embeddings endpoint. A local `all-MiniLM-L6-v2` model needs no paid service, keeps the documents on your machine during retrieval, and lets the tests run offline.

**Chunking.** Pages are split into 500-token windows with 50 tokens of overlap, counted with the embedding model's own tokenizer. One caveat: MiniLM truncates its input at 256 tokens, so with a 500-token chunk the second half is not reflected in the embedding. Ingestion logs a warning about this. The chunk size is configurable, and the eval can sweep it, so choose the value from measured results.

**Title in the embedding.** Each chunk is embedded as its page title plus its text, so a chunk that never names its product is still findable. The stored text and the snippets shown to users stay clean.

**Threshold.** The default 0.30 on cosine similarity is a starting value, not a tuned one. Too low and weak matches reach the LLM; too high and answerable questions are refused. Use the eval sweep to pick it for your docs.

**Exact search.** The index is a flat inner-product FAISS index over normalised vectors, which equals cosine similarity. For a few hundred chunks, approximate search would add complexity for no benefit.

**Grounding and citation checks.** The prompt restricts the model to the retrieved chunks, tells it to ignore instructions inside them, and requires JSON with `answerable`, `answer` and `used_chunk_ids`. Chunk IDs the model invents are dropped, and an answer that cites no retrieved chunk is rejected.

**Invalid output.** Output that is not valid JSON or fails validation gets one retry with a short correction, then falls back to `answered=false`.

**Failures.** Timeouts, 429s and 5xx responses from Groq are retried with exponential backoff and jitter, honouring `Retry-After`. After that the API returns a 503 with a clear message. Errors that retrying cannot fix, such as a bad key or model name, fail immediately.

**Input handling.** Questions are trimmed, whitespace is normalised, control characters are stripped, and empty or over-long input is rejected before any work happens.

## Limitations

- The rate limiter is in memory and per client IP: it resets on restart, is not shared across processes, and behind a reverse proxy all requests may share one IP.
- `confidence` is a retrieval similarity, not a probability of correctness.
- The eval checks that the right page was cited, not that the answer is correct. Read answers against their sources before relying on them.
- The labelled set is small, so rates move in large steps.
- Retrieval is dense-only; exact product names, versions and error codes can be missed.
- The index is a snapshot of the pages you saved. It does not update when the source pages change.
- Answers are only as good as the page text you saved. Pages that are mostly navigation or tables index poorly.
- One answer draws on at most `TOP_K` chunks, so questions that need information spread across many pages may be incomplete.

## Security

- Secrets come from environment variables or a git-ignored `.env`. `.env.example` contains no values.
- Retrieved text is treated as data, not instructions, in the prompt.
- Planned: a non-root Docker user, pinned dependencies in CI, CodeQL, and Dependabot.

## Project layout

```
app/         config, models, ingest, retriever, llm, ratelimit, main (FastAPI)
eval/        questions.jsonl, harness.py, run.py
tests/       offline unit and API tests with fakes
data/docs/   your documentation pages (front matter + text)
index/       generated FAISS index (git-ignored)
```

## What I would improve next

- Docker and Compose with a healthcheck, plus CI that runs lint, tests and a container build
- Hybrid retrieval (BM25 plus embeddings) and a reranker for exact terms
- A larger labelled set with held-out questions for the threshold choice
- Answer-correctness checks, not just citation checks
- Per-key rate limiting and request logging with correlation IDs
- Streaming responses and a minimal web UI
