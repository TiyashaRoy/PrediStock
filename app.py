import streamlit as st
import yfinance as yf
import plotly.express as px

st.set_page_config(
    page_title="PrediStock",
    page_icon="📈",
    layout="wide"
)

st.title("📈 PrediStock")
st.subheader("AI-Based Stock Market Analysis")

st.write(
    "Explore historical stock prices "
    "and compare multiple companies."
)

stocks = st.multiselect(
    "Select stocks",
    ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
    default=["AAPL", "GOOGL"]
)

if stocks:
    for stock in stocks:
        data = yf.Ticker(stock).history(period="1y")

        if not data.empty:
            st.subheader(stock)

            fig = px.line(
                data,
                x=data.index,
                y="Close",
                title=f"{stock} Stock Price"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.write("Latest stock data")
            st.dataframe(data.tail())
        else:
            st.warning(f"No data available for {stock}")

st.caption(
    "PrediStock | Educational stock market analysis"
)
