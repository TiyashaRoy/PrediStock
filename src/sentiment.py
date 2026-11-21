# PrediStock - News Sentiment Analysis

import yfinance as yf
import pandas as pd
import os

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# 1. Initialize sentiment analyzer

analyzer = SentimentIntensityAnalyzer()

# 2. Select stock

ticker_symbol = "AAPL"

stock = yf.Ticker(ticker_symbol)

# 3. Fetch financial news

news = stock.news

headlines = []

for item in news:
    content = item.get("content", {})
    title = content.get("title") or item.get("title")

    if title:
        headlines.append(title)

# 4. Check whether news is available

if not headlines:
    print("No financial news headlines found.")
    print("Please try again later.")

else:

    results = []

    # 5. Analyze each headline

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
            "Headline": headline,
            "Sentiment": sentiment,
            "Score": compound
        })

    # 6. Create DataFrame

    df = pd.DataFrame(results)

    # 7. Display results

    print("\nPREDISTOCK NEWS SENTIMENT ANALYSIS")
    print("----------------------------------")

    print(df.to_string(index=False))

    # 8. Count sentiment categories

    print("\nSENTIMENT SUMMARY")

    print(df["Sentiment"].value_counts())

    # 9. Save results

    os.makedirs("outputs", exist_ok=True)

    df.to_csv(
        "outputs/news_sentiment.csv",
        index=False
    )

    print("\nSentiment analysis completed!")
    print("Results saved in outputs/news_sentiment.csv")
