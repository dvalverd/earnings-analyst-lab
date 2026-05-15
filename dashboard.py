from dotenv import load_dotenv
load_dotenv()

import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
SEARCH_AVAILABLE = False
def query_transcript(query, ticker=None, n_results=5):
    return []

try:
    import importlib.util
    spec = importlib.util.find_spec("chromadb")
    if spec is not None:
        from embedder import query_transcript
        SEARCH_AVAILABLE = True
except Exception:
    pass

DATA_DIR = Path("data")

st.set_page_config(
    page_title="Earnings Signal Lab",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], p, div, span, label, input, button, select, textarea {
    font-family: 'Libre Baskerville', Georgia, serif !important;
}

.stApp {
    background-color: #f0ebf0;
    color: #1c1018;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none; }
[data-testid="stSidebar"] { display: none; }

.block-container {
    max-width: 1120px !important;
    padding: 3rem 2.5rem !important;
}

/* Page header */
.page-header {
    margin-bottom: 2.5rem;
    padding-bottom: 1.25rem;
    border-bottom: 2px solid #1c1018;
}
.page-title {
    font-size: 30px;
    font-weight: 700;
    color: #1c1018;
    letter-spacing: -0.01em;
    line-height: 1.1;
    margin-bottom: 5px;
}
.page-meta {
    font-size: 13px;
    font-weight: 400;
    color: #7a5f72;
    letter-spacing: 0.01em;
}

/* Section headers */
.section-head {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #7a5f72;
    margin: 2.5rem 0 1rem 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #d9ccd6;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #f7f2f6;
    border: 1px solid #d9ccd6;
    border-radius: 0;
    padding: 18px 22px !important;
}
[data-testid="stMetricLabel"] {
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: #7a5f72 !important;
    font-family: 'Libre Baskerville', Georgia, serif !important;
}
[data-testid="stMetricValue"] {
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #1c1018 !important;
    font-family: 'Libre Baskerville', Georgia, serif !important;
    letter-spacing: -0.02em !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 2px solid #1c1018;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    font-size: 12px;
    font-weight: 400;
    color: #7a5f72;
    padding: 10px 28px 10px 0;
    border: none;
    border-bottom: 2px solid transparent;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-family: 'Libre Baskerville', Georgia, serif !important;
}
.stTabs [aria-selected="true"] {
    color: #1c1018 !important;
    border-bottom: 2px solid #5e1f4a !important;
    background: transparent !important;
    font-weight: 700 !important;
}

/* Selectbox */
.stSelectbox label {
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: #7a5f72 !important;
}
.stSelectbox > div > div {
    background: #f7f2f6 !important;
    border: 1px solid #d9ccd6 !important;
    border-radius: 0 !important;
    font-size: 13px !important;
    color: #1c1018 !important;
}

/* Radio */
.stRadio > label {
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: #7a5f72 !important;
}
.stRadio label span {
    font-size: 13px !important;
    color: #1c1018 !important;
}

/* Text input */
.stTextInput label {
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: #7a5f72 !important;
}
.stTextInput input {
    background: #f7f2f6 !important;
    border: 1px solid #d9ccd6 !important;
    border-radius: 0 !important;
    font-size: 13px !important;
    color: #1c1018 !important;
    padding: 10px 12px !important;
    font-family: 'Libre Baskerville', Georgia, serif !important;
}
.stTextInput input:focus {
    border-color: #5e1f4a !important;
    box-shadow: 0 0 0 2px rgba(94,31,74,0.12) !important;
    outline: none !important;
}

/* Button */
.stButton button {
    background: #5e1f4a !important;
    color: #f0ebf0 !important;
    border: none !important;
    border-radius: 0 !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 10px 28px !important;
    font-family: 'Libre Baskerville', Georgia, serif !important;
}
.stButton button:hover {
    background: #7a2960 !important;
}

/* Download button */
.stDownloadButton button {
    background: transparent !important;
    color: #5e1f4a !important;
    border: 1px solid #d9ccd6 !important;
    border-radius: 0 !important;
    font-size: 11px !important;
    letter-spacing: 0.1em !important;
    font-family: 'Libre Baskerville', Georgia, serif !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: #f7f2f6 !important;
    border: 1px solid #d9ccd6 !important;
    border-radius: 0 !important;
    font-size: 13px !important;
    color: #1c1018 !important;
}
.streamlit-expanderContent {
    background: #f7f2f6 !important;
    border: 1px solid #d9ccd6 !important;
    border-top: none !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #d9ccd6 !important;
    border-radius: 0 !important;
}

/* Alerts */
.stAlert {
    background: #f7f2f6 !important;
    border: 1px solid #d9ccd6 !important;
    border-radius: 0 !important;
    font-size: 13px !important;
}
</style>
""", unsafe_allow_html=True)

C = {
    "bullish": "#3d7a5a",
    "neutral": "#9b8a94",
    "bearish": "#7a2940",
}

PLOT = dict(
    paper_bgcolor="#f0ebf0",
    plot_bgcolor="#f7f2f6",
    font=dict(family="Libre Baskerville, Georgia, serif", color="#1c1018", size=12),
    xaxis=dict(gridcolor="#e8dfe5", linecolor="#d9ccd6", tickfont=dict(color="#7a5f72", size=11)),
    yaxis=dict(gridcolor="#e8dfe5", linecolor="#d9ccd6", tickfont=dict(color="#7a5f72", size=11)),
    margin=dict(l=0, r=0, t=28, b=0),
)


@st.cache_data
def load_signals():
    path = DATA_DIR / "all_signals.json"
    if not path.exists():
        return pd.DataFrame()
    with open(path) as f:
        data = json.load(f)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_returns():
    path = DATA_DIR / "earnings_returns.json"
    if not path.exists():
        return pd.DataFrame()
    with open(path) as f:
        data = json.load(f)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    return df


@st.cache_data
def load_backtest():
    path = DATA_DIR / "backtest_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def join_data():
    signals = load_signals()
    returns = load_returns()
    if signals.empty or returns.empty:
        return pd.DataFrame()
    rows = []
    for _, sig in signals.iterrows():
        tr = returns[returns["ticker"] == sig["ticker"]].copy()
        if tr.empty:
            continue
        diffs = (tr["earnings_date"] - sig["date"]).abs()
        idx = diffs.idxmin()
        if diffs[idx].days <= 5:
            rows.append({**sig.to_dict(), **tr.loc[idx].to_dict()})
    return pd.DataFrame(rows)


st.markdown("""
<div class="page-header">
    <div class="page-title">Earnings Signal Lab</div>
    <div class="page-meta">LLM-extracted sentiment signals benchmarked against forward price returns</div>
</div>
""", unsafe_allow_html=True)

df_full = join_data()

col_f1, col_f2, col_f3 = st.columns([2, 3, 5])
with col_f1:
    if not df_full.empty:
        tickers = ["All"] + sorted(df_full["ticker"].unique().tolist())
        selected_ticker = st.selectbox("Ticker", tickers)
    else:
        selected_ticker = "All"
        st.selectbox("Ticker", ["All"], disabled=True)

with col_f2:
    return_window = st.radio(
        "Return window",
        ["return_1d", "return_5d"],
        format_func=lambda x: "1-day" if x == "return_1d" else "5-day",
        horizontal=True,
    )

df = df_full[df_full["ticker"] == selected_ticker] if (not df_full.empty and selected_ticker != "All") else df_full

if df_full.empty:
    st.warning("No matched data. Run ingestor, embedder, extractor, backtester first.")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Signal Alpha", "Transcript Search", "Raw Data"])

with tab1:
    backtest = load_backtest()
    overall = backtest.get("overall", {})

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Matched Events", overall.get("total_events", len(df)))
    with c2:
        v = overall.get("bullish_avg_5d")
        st.metric("Bullish 5d Avg", f"{v:.2%}" if v is not None else "n/a")
    with c3:
        v = overall.get("bearish_avg_5d")
        st.metric("Bearish 5d Avg", f"{v:.2%}" if v is not None else "n/a")
    with c4:
        v = overall.get("long_short_sharpe")
        st.metric("L/S Sharpe", f"{v:.2f}" if v is not None else "n/a")

    if not df.empty and "overall_sentiment" in df.columns:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("<div class='section-head'>Sentiment distribution</div>", unsafe_allow_html=True)
            counts = df["overall_sentiment"].value_counts().reset_index()
            counts.columns = ["sentiment", "n"]
            fig = px.bar(counts, x="sentiment", y="n", color="sentiment", color_discrete_map=C)
            fig.update_layout(**PLOT, showlegend=False, height=260)
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown("<div class='section-head'>Return distribution by sentiment</div>", unsafe_allow_html=True)
            fig2 = px.box(df, x="overall_sentiment", y=return_window,
                          color="overall_sentiment", color_discrete_map=C,
                          labels={return_window: "Forward return"})
            fig2.update_layout(**PLOT, showlegend=False, height=260)
            st.plotly_chart(fig2, use_container_width=True)

        if "management_tone" in df.columns:
            st.markdown("<div class='section-head'>Average return by management tone</div>", unsafe_allow_html=True)
            tone_df = df.groupby("management_tone")[return_window].mean().reset_index()
            tone_df.columns = ["tone", "avg"]
            tone_df = tone_df.sort_values("avg")
            fig3 = go.Figure(go.Bar(
                x=tone_df["avg"], y=tone_df["tone"],
                orientation="h",
                marker_color=[C["bullish"] if v >= 0 else C["bearish"] for v in tone_df["avg"]],
                marker_line_width=0,
                text=[f"{v:.2%}" for v in tone_df["avg"]],
                textposition="outside",
                textfont=dict(color="#7a5f72", size=11),
            ))
            fig3.update_layout(**PLOT, height=200, xaxis_tickformat=".1%")
            st.plotly_chart(fig3, use_container_width=True)

with tab2:
    if df.empty:
        st.info("No data available.")
    else:
        signal_cols = [s for s in ["overall_sentiment", "management_tone", "guidance_direction",
                                    "revenue_surprise_language", "eps_surprise_language"] if s in df.columns]
        col_pick, _ = st.columns([1, 3])
        with col_pick:
            selected_signal = st.selectbox("Signal type", signal_cols)

        alpha_df = (
            df.groupby(selected_signal)[return_window]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": "avg_return", "std": "std_return", "count": "n"})
        )
        alpha_df["hit_rate"] = df.groupby(selected_signal)[return_window].apply(
            lambda x: (x > 0).mean()
        ).values
        alpha_df = alpha_df.sort_values("avg_return", ascending=False)

        col_chart, col_tbl = st.columns([3, 2])
        with col_chart:
            st.markdown("<div class='section-head'>Average forward return by signal value</div>", unsafe_allow_html=True)
            fig4 = go.Figure(go.Bar(
                x=alpha_df[selected_signal],
                y=alpha_df["avg_return"],
                marker_color=[C["bullish"] if v >= 0 else C["bearish"] for v in alpha_df["avg_return"]],
                marker_line_width=0,
                text=[f"n={n}" for n in alpha_df["n"]],
                textposition="outside",
                textfont=dict(color="#7a5f72", size=11),
                error_y=dict(array=alpha_df["std_return"], color="#c8b8c4", thickness=1.5),
            ))
            fig4.update_layout(**PLOT, height=320, yaxis_tickformat=".1%")
            st.plotly_chart(fig4, use_container_width=True)

        with col_tbl:
            st.markdown("<div class='section-head'>Statistics</div>", unsafe_allow_html=True)
            out = alpha_df[[selected_signal, "n", "avg_return", "hit_rate"]].copy()
            out["avg_return"] = out["avg_return"].map("{:.2%}".format)
            out["hit_rate"] = out["hit_rate"].map("{:.0%}".format)
            out.columns = ["Signal", "N", "Avg Return", "Hit Rate"]
            st.dataframe(out, use_container_width=True, hide_index=True)

        if "sentiment_score" in df.columns:
            st.markdown("<div class='section-head'>Information coefficient, sentiment score vs forward return</div>", unsafe_allow_html=True)
            fig5 = px.scatter(
                df, x="sentiment_score", y=return_window,
                color="overall_sentiment", color_discrete_map=C,
                trendline="ols",
                hover_data=["ticker", "date"],
                labels={return_window: "Forward return", "sentiment_score": "Sentiment score"},
            )
            fig5.update_layout(**PLOT, height=360)
            fig5.update_traces(marker=dict(size=7, opacity=0.6, line=dict(width=0)))
            st.plotly_chart(fig5, use_container_width=True)

with tab3:
    st.markdown("<div class='section-head'>About this search</div>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size:13px;color:#7a5f72;line-height:1.8;margin-bottom:1.5rem'>
    Type any concept, a business theme, a risk, or a financial metric, and this returns
    the most semantically relevant passages from all ingested earnings call transcripts.
    Results are ranked by vector similarity, not keyword matching.
    </p>
    """, unsafe_allow_html=True)

    col_q, col_t = st.columns([3, 1])
    with col_q:
        query = st.text_input("Search query", placeholder="e.g. supply chain headwinds impacting margins")
    with col_t:
        search_ticker = st.text_input("Filter by ticker", placeholder="e.g. AAPL")

    col_btn, col_n, _ = st.columns([1, 1, 4])
    with col_btn:
        run_search = st.button("Search transcripts")
    with col_n:
        n_results = st.selectbox("Results", [5, 10, 15])

    if not SEARCH_AVAILABLE:
        st.info("Transcript search is available when running locally. The vector database requires ChromaDB which is not supported on Python 3.14. Run the app locally with python3 -m streamlit run dashboard.py to use this feature.")
    elif run_search and query:
        try:
            ticker_filter = search_ticker.strip().upper() or None
            results = query_transcript(query, ticker=ticker_filter, n_results=n_results)
            if not results:
                st.info("No results found. Make sure embedder.py has been run.")
            else:
                st.markdown(f"<div class='section-head'>{len(results)} results</div>", unsafe_allow_html=True)
                for r in results:
                    meta = r["metadata"]
                    score = round((1 - r["distance"]) * 100, 1)
                    with st.expander(f"{meta['ticker']}   {meta['date']}   {score}% similarity"):
                        st.markdown(f"<p style='font-size:13px;color:#1c1018;line-height:1.8'>{r['text']}</p>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Search error: {e}")

with tab4:
    if df.empty:
        st.info("No data available.")
    else:
        st.markdown("<div class='section-head'>Joined signals and returns</div>", unsafe_allow_html=True)
        display_cols = [c for c in ["ticker", "date", "overall_sentiment", "sentiment_score",
                                     "management_tone", "guidance_direction",
                                     "return_1d", "return_5d"] if c in df.columns]
        out = df[display_cols].copy()
        for col in ["return_1d", "return_5d"]:
            if col in out.columns:
                out[col] = out[col].map("{:.2%}".format)
        if "sentiment_score" in out.columns:
            out["sentiment_score"] = out["sentiment_score"].map("{:.2f}".format)
        st.dataframe(out, use_container_width=True, hide_index=True)
        csv = df[display_cols].to_csv(index=False)
        st.download_button("Export CSV", csv, "earnings_signals.csv", "text/csv")
