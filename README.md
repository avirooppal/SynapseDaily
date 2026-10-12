# AI Digest → Telegram

Every morning (08:00 IST) this pulls the day's AI news and papers, merges duplicates across sources,
ranks them, reads the actual pages, and sends you short grounded summaries on Telegram.
Cost: free (GitHub Actions + Gemini free tier + Telegram).

Sources: Hacker News, HF Daily Papers, arXiv (cs.AI/LG/CL), r/LocalLLaMA, r/MachineLearning,
official lab blogs, and new model repos from DeepSeek/Qwen/Meta/Mistral/Google/etc. on Hugging Face.

## Setup (10 minutes)

1. **Telegram bot**: message `@BotFather` → `/newbot` → copy the token.
   Then send any message to your new bot, open
   `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy `message.chat.id`.
2. **Gemini key**: https://aistudio.google.com → *Get API key* (free, no billing).
3. **GitHub**: push this folder to a new repo (private is fine), then
   *Settings → Secrets and variables → Actions* and add:
   `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `GEMINI_API_KEY`.
4. **Test**: *Actions → Daily AI Digest → Run workflow*. After that it runs daily on its own.

Local dry run (prints instead of sending, doesn't touch `seen.json`):

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...
python digest.py --dry-run
```

## Tuning (all at the top of `digest.py`)

| What | Where |
|---|---|
| Items per day | `TOP_N` (or env var) |
| Add/remove blogs | `LAB_FEEDS` (any RSS/Atom URL) |
| Which labs count as "model releases" | `HF_ORGS` |
| HN topic filter | `AI_WORDS` |
| Mix of papers vs models vs news | `KIND_CAPS` |
| Send time | cron in `.github/workflows/daily-digest.yml` (UTC) |
| Gemini model | env `GEMINI_MODEL` (default: `gemini-flash-latest`, falls back to flash-lite) |

## Notes

- Summaries use only the fetched page/abstract/model card (+ `config.json` for model repos), and the
  model may mark items as `skip` if there's nothing concrete. Always click through for anything important.
- Reddit sometimes blocks GitHub's IPs; that source is then skipped silently and the rest still runs.
- Anthropic has no RSS feed I could confirm, so it isn't in `LAB_FEEDS`; add one if you find a feed you trust.
- If GitHub pauses the schedule after a long idle period, re-enable it in the Actions tab.
- Offline tests: `python tests/test_digest.py`
