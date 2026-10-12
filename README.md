# SynapseDaily

SynapseDaily is an automated intelligence pipeline that aggregates, deduplicates, ranks, and summarizes daily artificial intelligence research, open-source model releases, and technical news. Summaries are delivered daily directly to Telegram.

The entire stack is designed to operate within free operational tiers (GitHub Actions, OpenRouter / Gemini free tiers, and Telegram Bot API).

**Tracked Sources:**
- Academic Preprints: arXiv (`cs.AI`, `cs.LG`, `cs.CL`), Hugging Face Daily Papers
- Model Hubs: Hugging Face release feeds for primary organizations (DeepSeek, Qwen, Meta, Mistral, Google, etc.)
- Industry Research Labs: Official engineering blogs (Google DeepMind, OpenAI, NVIDIA, Microsoft Research, Hugging Face)
- Community Aggregators: Hacker News, Reddit (`r/LocalLLaMA`, `r/MachineLearning`)

---

## Architecture and Pipeline

1. **Collection:** Concurrently fetches recent items published across all tracked endpoints within a configurable lookback window.
2. **Deduplication:** Normalizes URLs and applies fuzzy string distance matching to collapse cross-posted links into single story representations.
3. **Ranking and Filtering:** Scores stories by community traction and cross-source corroboration while enforcing category caps to ensure a balanced mix of papers, model releases, and news.
4. **Extraction and Grounding:** Resolves underlying article text, arXiv abstracts, or Hugging Face model cards (`README.md` and `config.json`).
5. **Summarization:** Uses structured LLM generation with strict schema enforcement to generate concise technical bullet points, relevance context, and priority scores based strictly on source text.
6. **Dispatch and State Persistence:** Delivers formatted HTML messages to Telegram and commits delivered IDs to `seen.json` to prevent re-delivery over a 30-day window.

---

## Setup Guide

### 1. Telegram Bot Configuration
1. Message `@BotFather` on Telegram, create a new bot using `/newbot`, and save the API token.
2. Start a conversation with your bot by sending any message.
3. Query `https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/getUpdates` to locate your numeric `message.chat.id`.

### 2. Large Language Model Provider
- **OpenRouter (Default):** Generate an API key at [openrouter.ai/keys](https://openrouter.ai/keys). Defaults to `nvidia/nemotron-3-ultra-550b-a55b:free`.
- **Google Gemini (Alternative):** Generate an API key at [Google AI Studio](https://aistudio.google.com).

### 3. GitHub Actions Configuration
Configure the following secrets in your repository under **Settings > Secrets and variables > Actions > Secrets**:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `OPENROUTER_API_KEY` (or `GEMINI_API_KEY`)

Optional repository variables can be defined under **Actions > Variables**:
- `OPENROUTER_MODEL`: Model identifier (defaults to `nvidia/nemotron-3-ultra-550b-a55b:free`).

### 4. Verification
1. Navigate to the **Actions** tab in GitHub.
2. Select **Daily AI Digest** and execute via **Run workflow**.
3. The workflow runs on a daily schedule at 08:00 IST (02:30 UTC).

---

## Local Development and Dry Run

To execute the aggregation and summarization pipeline locally without sending messages or updating `seen.json`:

```bash
pip install -r requirements.txt

# Run with OpenRouter
export OPENROUTER_API_KEY="sk-or-v1-..."
python digest.py --dry-run

# Run with Gemini
export GEMINI_API_KEY="..."
python digest.py --dry-run
```

---

## Configuration Reference

Key parameters can be configured either at the top of `digest.py` or via environment variables:

| Variable | Default Value | Description |
|---|---|---|
| `TOP_N` | `8` | Maximum number of items included in each daily digest |
| `LLM_PROVIDER` | Auto | Provider selection (`openrouter` or `gemini`) |
| `OPENROUTER_MODEL` | `nvidia/nemotron-3-ultra-550b-a55b:free` | Model identifier used for OpenRouter requests |
| `GEMINI_MODEL` | `gemini-flash-latest` | Model identifier used for Gemini requests |
| `TZ_NAME` | `Asia/Kolkata` | Timezone identifier for timestamp generation |
| `KIND_CAPS` | Paper: 4, Model: 3, Disc: 2 | Category distribution limits |
| `SEEN_TTL_DAYS` | `30` | Retention window for deduplication history |

---

## Delivery Format Sample

```text
SynapseDaily - Mon 12 Oct
5 items - HN, arXiv, HF, Reddit, lab blogs
------------------------------------------------------------
1. DeepSeek Releases V3 Mixture-of-Experts Architecture

- 671B total parameters with 37B activated per token across 256 routed experts
- Multi-head Latent Attention (MLA) reduces KV cache memory footprint by over 90%
- DualPipe FP8 training framework enables overlap of computation and communication
- Practical Significance: Matches dense frontier model benchmarks with substantially lower inference resource requirements.
- Links: Paper (arXiv) | Source (Hugging Face)
```

---

## Quality and Verification

- **Strict Grounding:** The prompt forbids speculation, synthetic benchmark extrapolation, and unverified parameter sizing. Items lacking technical depth are filtered out.
- **Automated Tests:** Execute offline test suites locally using:
  ```bash
  python tests/test_digest.py
  ```
