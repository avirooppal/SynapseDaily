# 📡 AI Digest → Telegram

Every morning (08:00 IST) this pulls the day's AI news, research papers, and model releases, merges duplicates across sources, ranks them, reads the underlying pages, and sends grounded technical summaries directly to your Telegram.

**Cost:** $0 / Free (GitHub Actions + OpenRouter / Gemini free tiers + Telegram Bot API).

**Sources:** Hacker News, Hugging Face Daily Papers, arXiv (`cs.AI`, `cs.LG`, `cs.CL`), Reddit (`r/LocalLLaMA`, `r/MachineLearning`), official lab blogs (DeepMind, OpenAI, NVIDIA, Hugging Face, Microsoft), and fresh model weights on Hugging Face (DeepSeek, Qwen, Meta, Mistral, Google, etc.).

---

## ⚡ Setup (10 minutes)

### 1. Telegram Bot
- Open Telegram and message `@BotFather` → `/newbot` → follow instructions and copy your **Bot Token**.
- Send any message to your new bot.
- Open `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in your browser and copy `message.chat.id` (your **Chat ID**).

### 2. LLM API Key (Free)
- **OpenRouter** (Default & Recommended): Create a free key at [openrouter.ai/keys](https://openrouter.ai/keys).
  - Defaults to `nvidia/nemotron-3-ultra-550b-a55b:free` (no billing needed).
  - Or use any OpenRouter model (e.g. `meta-llama/llama-3.3-70b-instruct:free`).
- **Gemini** (Alternative): Get a free API key at [Google AI Studio](https://aistudio.google.com).

### 3. GitHub Repository Secrets
Push this folder to a GitHub repository (private or public), then go to **Settings → Secrets and variables → Actions** and add:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `OPENROUTER_API_KEY` *(or `GEMINI_API_KEY`)*

*(Optional)* Under **Variables**, add `OPENROUTER_MODEL` if you want to switch from the default Nemotron model to another model.

### 4. Test Run
- Go to the **Actions** tab in your GitHub repository.
- Select **Daily AI Digest** → Click **Run workflow**.
- After this test run, GitHub Actions will trigger it automatically every day at 08:00 IST (02:30 UTC).

---

## 🧪 Local Dry Run

Test locally without sending messages or modifying deduplication state (`seen.json`):

```bash
pip install -r requirements.txt

# Run with OpenRouter (recommended)
export OPENROUTER_API_KEY="sk-or-v1-..."
python digest.py --dry-run

# Or run with Gemini
export GEMINI_API_KEY="..."
python digest.py --dry-run
```

---

## ⚙️ Tuning (Configured at top of `digest.py` or via Env Vars)

| What | Setting | Default | Description |
|---|---|---|---|
| Items per digest | `TOP_N` | `8` | Number of top ranked items delivered |
| LLM Provider | `LLM_PROVIDER` | Auto | `openrouter` or `gemini` (auto-detected from keys) |
| OpenRouter Model | `OPENROUTER_MODEL` | `nvidia/nemotron-3-ultra-550b-a55b:free` | Model ID on OpenRouter |
| Gemini Model | `GEMINI_MODEL` | `gemini-flash-latest` | Model ID on Google AI Studio |
| Timezone | `TZ_NAME` | `Asia/Kolkata` | Timezone for header date |
| Lab Feeds | `LAB_FEEDS` | In code | RSS/Atom feeds for official AI labs |
| Tracked HF Orgs | `HF_ORGS` | In code | Organizations tracked for model weight releases |
| Content Mix | `KIND_CAPS` | Paper: 4, Model: 3, Disc: 2 | Cap per category to balance diversity |
| Schedule Time | Cron in `.github/workflows/daily-digest.yml` | `30 2 * * *` (02:30 UTC / 08:00 IST) | Execution cron expression |

---

## 📋 Example Telegram Output

> **📡 AI Digest — Mon 12 Oct**  
> *5 items · HN, arXiv, HF, Reddit, lab blogs*
>
> **1. DeepSeek releases V3 Mixture-of-Experts Architecture**  
> • Total 671B parameters with 37B activated per token across 256 routed experts  
> • Multi-head Latent Attention (MLA) reduces KV cache footprint by over 90%  
> • FP8 mixed-precision training framework scales throughput to 14.8M tokens/sec  
> 💡 *Significant reduction in inference VRAM requirements while matching frontier benchmark performance.*  
> 🔗 [Read](https://arxiv.org/abs/...) · [HN](https://news.ycombinator.com/...) · [HF Hub](https://huggingface.co/...)

---

## 💡 Notes

- **Grounded Summaries:** The LLM only summarizes facts extracted from the fetched article text, abstract, or model repository card. If an article lacks concrete details, it is discarded.
- **Deduplication:** `seen.json` tracks items delivered over a 30-day window to prevent repetitive stories across days.
- **Offline Tests:** Run `python tests/test_digest.py` to run offline unit tests without network calls.
