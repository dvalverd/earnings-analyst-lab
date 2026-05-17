# Earnings Signal Lab

A pipeline that extracts structured signals from earnings call transcripts using Claude, then backtests those signals against forward price returns. Includes a Streamlit dashboard for exploring results.

<kbd>[▸ Live Demo — earnings-analyst-lab.onrender.com](https://earnings-analyst-lab.onrender.com)</kbd>
## What it does

1. Loads real earnings call transcripts from a Kaggle dataset
2. Downloads historical price data for each ticker
3. Embeds transcripts into a vector database using ChromaDB
4. Sends each transcript to Claude to extract structured signals: sentiment, management tone, guidance direction, revenue and EPS surprise language
5. Joins signals to forward price returns and computes alpha, hit rate, Sharpe ratio, and information coefficient by signal type
6. Displays results in an interactive dashboard with semantic transcript search

## Stack

Python, Anthropic Claude API, ChromaDB, yfinance, Streamlit, Plotly

## Setup

Install dependencies:

```bash
pip3 install -r requirements.txt
```

Copy the env file and add your Anthropic API key:

```bash
cp .env
```

Open `.env` and set:

```
ANTHROPIC_API_KEY=sk..........
```

Get your key from the console dashboard

## Data

Download the Motley Fool earnings call transcript dataset from Kaggle:

https://www.kaggle.com/datasets/tpotterer/motley-fool-scraped-earnings-call-transcripts

Place the `.pkl` file in the project root.

## Run order

```bash
python3 load_kaggle.py --file motley-fool-data.pkl
python3 ingestor.py
python3 embedder.py
python3 extractor.py
python3 backtester.py
python3 -m streamlit run dashboard.py
```

Steps 1 through 5 only need to be run once. After that, only step 6 is needed to open the dashboard.

## Extracted signals

For each earnings call, Claude extracts:

- `overall_sentiment`: bullish, neutral, or bearish
- `sentiment_score`: float from -1.0 to 1.0
- `management_tone`: confident, cautious, defensive, or hedging
- `guidance_direction`: raised, maintained, lowered, or withdrawn
- `revenue_surprise_language`: beat, missed, or in-line
- `eps_surprise_language`: beat, missed, or in-line
- `key_bullish_phrases`: direct quotes from the transcript
- `key_bearish_phrases`: direct quotes from the transcript
- `macro_concerns_mentioned`: list of macro risks mentioned

## Backtest metrics

- Average forward return per signal value, 1-day and 5-day
- Hit rate: percentage of events where signal correctly predicted direction
- Information coefficient: Spearman rank correlation between sentiment score and forward return
- Long/short annualized Sharpe ratio: long bullish calls, short bearish calls

## Dashboard tabs

- Overview: sentiment distribution, return distribution by sentiment, return by management tone
- Signal Alpha: average return by signal value, statistics table, IC scatter plot
- Transcript Search: semantic search across all ingested transcripts using vector similarity
- Raw Data: full joined table of signals and returns, exportable as CSV
