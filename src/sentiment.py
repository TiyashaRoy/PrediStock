# PrediStock - News Sentiment Analysis
# Using Google News RSS instead of Yahoo Finance

import feedparser
import pandas as pd
import os

from urllib.parse import quote_plus
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# 1. Initialize analyzer

analyzer = SentimentIntensityAnalyzer()

# 2. Select stock

ticker_symbol = "AAPL"

# 3. Fetch news from Google News RSS

query = quote_plus(f"{ticker_symbol} stock")

url = (
    f"https://news.google.com/rss/search?q={query}"
    "&hl=en-US&gl=US&ceid=US:en"
)

feed = feedparser.parse(url)

headlines = [
    entry.title for entry in feed.entries
    if entry.get("title")
]

# 4. Check news availability

if not headlines:
    print("No news headlines found.")
    print("Please try again later.")

else:

    results = []

    # 5. Analyze headlines

    for headline in headlines:

        scores = analyzer.polarity_scores(headline)

        compound = scores["compound"]

        if compound >= 0.05:
            sentiment = "Positive"

        elif compound <= -0.05:
            sentiment = "Negative"

        else:
            sentiment = "Neutral"

        results.append({
            "Stock": ticker_symbol,
            "Headline": headline,
            "Sentiment": sentiment,
            "Score": compound
        })

    # 6. Create DataFrame

    df = pd.DataFrame(results)

    print("\nPREDISTOCK NEWS SENTIMENT ANALYSIS")
    print("----------------------------------")

    print(df.to_string(index=False))

    # 7. Sentiment summary

    print("\nSENTIMENT SUMMARY")

    print(df["Sentiment"].value_counts())

    # 8. Save results

    os.makedirs("outputs", exist_ok=True)

    df.to_csv(
        "outputs/news_sentiment.csv",
        index=False
    )

    print("\nSentiment analysis completed!")
    print("Saved: outputs/news_sentiment.csv")
