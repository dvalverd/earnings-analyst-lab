
import os
from dotenv import load_dotenv
load_dotenv()
import json
import anthropic
from pathlib import Path
from embedder import query_transcript, get_collection

DATA_DIR = Path("data")
SIGNALS_DIR = DATA_DIR / "signals"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

EXTRACTION_PROMPT = """You are a quantitative analyst specializing in earnings call analysis.

Below is an excerpt from an earnings call transcript for {ticker} on {date}.

Extract the following signals and return ONLY valid JSON -no explanation, no markdown.

{{
  "ticker": "{ticker}",
  "date": "{date}",
  "overall_sentiment": "bullish" | "neutral" | "bearish",
  "sentiment_score": float between -1.0 (very bearish) and 1.0 (very bullish),
  "management_tone": "confident" | "cautious" | "defensive" | "hedging",
  "guidance_direction": "raised" | "maintained" | "lowered" | "withdrawn" | "none",
  "revenue_surprise_language": "beat" | "missed" | "in-line" | "none",
  "eps_surprise_language": "beat" | "missed" | "in-line" | "none",
  "key_bullish_phrases": [list of up to 3 direct quotes that are bullish],
  "key_bearish_phrases": [list of up to 3 direct quotes that are bearish or cautionary],
  "macro_concerns_mentioned": [list of macro risks mentioned, e.g. "interest rates", "FX headwinds"],
  "confidence": float between 0.0 and 1.0 indicating how confident you are in this extraction
}}

TRANSCRIPT EXCERPT:
{transcript}
"""


def extract_signals_from_text(ticker: str, date: str, transcript_text: str) -> dict:
    
    prompt = EXTRACTION_PROMPT.format(
        ticker=ticker,
        date=date,
        transcript=transcript_text[:6000],  # stay within context limits
    )

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    signals = json.loads(raw)
    return signals


def extract_signals_via_rag(ticker: str, date: str) -> dict:
 
    queries = [
        "revenue earnings guidance outlook next quarter",
        "risks challenges headwinds concerns",
        "growth expansion opportunities bullish",
        "management tone confident cautious defensive",
    ]

   
    seen = set()
    chunks = []
    for q in queries:
        results = query_transcript(q, ticker=ticker, n_results=3)
        for r in results:
            chunk_id = r["metadata"].get("chunk_index", "")
            key = f"{ticker}_{date}_{chunk_id}"
            if key not in seen:
                seen.add(key)
                chunks.append(r["text"])

    combined = "\n\n---\n\n".join(chunks[:8])  # top 8 
    return extract_signals_from_text(ticker, date, combined)


def process_all_transcripts() -> list[dict]:
    
    transcript_files = list(TRANSCRIPTS_DIR.glob("*.txt"))
    all_signals = []

    for path in transcript_files:
        name = path.stem
        parts = name.split("_", 1)
        if len(parts) != 2:
            continue
        ticker, date = parts

        out_path = SIGNALS_DIR / f"{name}.json"
        if out_path.exists():
            print(f"  Already extracted: {name}, ")
            with open(out_path) as f:
                all_signals.append(json.load(f))
            continue

        print(f"  Extracting signals: {ticker} {date}..")
        try:
            text = path.read_text(encoding="utf-8")
            signals = extract_signals_from_text(ticker, date, text)
            with open(out_path, "w") as f:
                json.dump(signals, f, indent=2)
            all_signals.append(signals)
            print(f"   sentiment={signals.get('overall_sentiment')} "
                  f"tone={signals.get('management_tone')} "
                  f"guidance={signals.get('guidance_direction')}")
        except Exception as e:
            print(f"    ERROR: {e}")

    # Save combined
    combined_path = DATA_DIR / "all_signals.json"
    with open(combined_path, "w") as f:
        json.dump(all_signals, f, indent=2)
    print(f"\nAll signals saved : {combined_path}")
    return all_signals


def print_signal_summary(signals: list[dict]):
    from collections import Counter
    sentiments = Counter(s.get("overall_sentiment") for s in signals)
    tones = Counter(s.get("management_tone") for s in signals)
    guidance = Counter(s.get("guidance_direction") for s in signals)

    print("\nSignal Summary ")
    print(f"Total extracted: {len(signals)}")
    print(f"Sentiment:  {dict(sentiments)}")
    print(f"Tone:       {dict(tones)}")
    print(f"Guidance:   {dict(guidance)}")


if __name__ == "__main__":
    signals = process_all_transcripts()
    if signals:
        print_signal_summary(signals)
    else:
        print("No transcripts found. Add .txt files to data/transcripts/ first.")
        print("Format: TICKER_YYYY-MM-DD.txt (e.g. AAPL_2024-01-25.txt)")
