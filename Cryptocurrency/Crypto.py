import streamlit as st
from PIL import Image
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from bs4 import BeautifulSoup
import requests
import json
import time

st.set_page_config(layout="wide")   #st.beta_set_page_config(layout="wide")

img = Image.open('Crypto_Header.png')

st.image(img, width = 500)
st.title("Crypto Price app")
st.markdown('''
This app retrieves cypto prices for the top 100 crypto currencies
''')

bar_expand = st.beta_expander("About")
bar_expand.markdown('''
**Data Source :** https://coinmarketcap.com/ [CoinMarketCap]
''')

col1 = st.sidebar
col2, col3 = st.beta_columns((2,1))

col1.header("Input Value")
curr_per_unit = col1.selectbox("Select the currency", ("USD", "BTC", "ETH"))

@st.cache
def load_data():
    req = requests.get("https://coinmarketcap.com/")
    soup = BeautifulSoup(req.content, "html.parser")
    page_data = soup.find('script', id='__NEXT_DATA__', type='application/json')
    coins = {}
    coin_data = json.loads(page_data.contents[0])
    listing = coin_data['props']['initialState']['cryptocurrency']['listingLatest']['data']
    for i in listing:
        coins[str(i['id'])] = i['slug']

    coin_name = []
    coin_symbol = []
    market_cap = []
    percent_change_1h = []
    percent_change_24h = []
    percent_change_7d = []
    price = []
    volume_24h = []

    for i in listing:
        coin_name.append(i['slug'])
        coin_symbol.append(i['symbol'])
        market_cap.append(i['quote'][curr_per_unit]['price'])
        percent_change_1h.append(i['quote'][curr_per_unit]['percentChange1h'])
        percent_change_24h.append(i['quote'][curr_per_unit]['percentChange24h'])
        percent_change_7d.append(i['quote'][curr_per_unit]['percentChange7d'])
        price.append(i['quote'][curr_per_unit]['marketCap'])
        volume_24h.append(i['quote'][curr_per_unit]['volume24h'])

    df = pd.DataFrame(columns=['coin_name', 'coin_symbol', 'Price', 'percentChange1h', 'percentChange24h', 'percentChange7d', 'marketCap', 'volume24h'])
    df['coin_name'] = coin_name
    df['coin_symbol'] = coin_symbol
    df['Price'] = price
    df['percentChange1h'] = percent_change_1h
    df['percentChange24h'] = percent_change_24h
    df['percentChange7d'] = percent_change_7d
    df['marketCap'] = market_cap
    df['volume24h'] = volume_24h
    return df

dataframe = load_data()

sort_coin = sorted(dataframe['coin_symbol'])

select_coin = col1.multiselect("Crptocurrency", sort_coin,sort_coin)
df_select_coin = dataframe[(dataframe['coin_symbol'].isin(select_coin))]

num_coin = col1.slider("Display Top N coins", 1, 100, 100)
df_coin = df_select_coin[:num_coin]

percent_timeframe = col1.selectbox("Percent change timeframe", ["7d", "24h", "1h"])
percent_dict = {"7d":"percentChange7d","24h":"percentChange24h","1h":"percentChange1h"}

select_timeframe = percent_dict[percent_timeframe]

sort_value = col1.selectbox("Sort Values", ["Yes", "No"])

col2.subheader("Price Data of selected crypto")
col2.write("Data Dimension: " +str(df_select_coin.shape[0])+ " rows and " +str(df_select_coin.shape[1])+ "column")

col2.dataframe(df_coin)

def download(dat):
    csv = dat.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="crypto.csv"> Download File </a>'
    return href

col2.markdown(download(df_select_coin),unsafe_allow_html=True)


col2.subheader("Table of percent price change")
df_change = pd.concat([df_coin.coin_symbol, df_coin.percentChange1h, df_coin.percentChange24h, df_coin.percentChange7d], axis=1)
df_change = df_change.set_index('coin_symbol')
df_change['positive_percent_change_1h'] = df_change["percentChange1h"] > 0
df_change['positive_percent_change_24h'] = df_change["percentChange24h"] > 0
df_change['positive_percent_change_7d'] = df_change["percentChange7d"] > 0
col2.dataframe(df_change)

col3.subheader("Bar plot of % price change")

if percent_timeframe == "7d":
    if sort_value == "Yes":
        df_change = df_change.sort_values(by=['percentChange7d'])
    col3.write('*7 days period*')
    plt.figure(figsize=(5,25))
    plt.subplots_adjust(top = 1, bottom = 0)
    df_change['percentChange7d'].plot(kind='barh', color = df_change.positive_percent_change_7d.map({True:'g', False:'r' }))
    col3.pyplot(plt)

elif percent_timeframe == "24h":
    if sort_value == "Yes":
        df_change = df_change.sort_values(by=['percentChange24h'])
    col3.write('*24 hours period*')
    plt.figure(figsize=(5,25))
    plt.subplot_adjust(top = 1, bottom = 0)
    df_change['percentChange24h'].plot(kind='barh', color = df_change.positive_percent_change_24h.map({True:'g', False:'r'}))
    col3.pyplot(plt)

elif percent_timeframe == "1h":
    if sort_value == "Yes":
        df_change = df_change.sort_values(by=['percentChange1h'])
    col3.write('*1 hour period*')
    plt.figure(figsize=(5,25))
    plt.subplot_adjust(top = 1, bottom = 0)
    df_change['percentChange1h'].plot(kind='barh', color = df_change.positive_percent_change_1h.map({True:'g', False:'r'}))
    col3.pyplot(plt)

