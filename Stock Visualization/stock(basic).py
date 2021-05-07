import yfinance as yf
import streamlit as st
import pandas as pd

st.write("""
# Simple Stock price app
Shown are the stock **closing price** and **volume** of google!

""")        #learn markdown to do more

tickersym = "GME"

tickerdat = yf.Ticker(tickersym)

tickergraph = tickerdat.history(period="1d", start="2020-12-1", end="2021-3-1")

st.write('''# Closing Price''')
st.line_chart(tickergraph.Close)
st.write('''# Volume''')
st.line_chart(tickergraph.Volume)
