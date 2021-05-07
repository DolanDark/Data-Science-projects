import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import base64
#import seaborn as sns
import yfinance as yf

st.title("S&P 500 app")

st.markdown('''
This app retrieves te list of S&P 500 companies and tablulates their data
''')

@st.cache
def load_data():
    wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = pd.read_html(wiki_url, header=0)
    dataframe = html[0]
    return dataframe

dataframe = load_data()

unique_sector = sorted(dataframe['GICS Sector'].unique())
selected_sector = st.sidebar.multiselect("Sector ", unique_sector, unique_sector)

df_selected_sector = dataframe[ (dataframe['GICS Sector'].isin(selected_sector))]

st.header('Display companies in a selected sector')
st.write("Data Dimension: " + str(df_selected_sector.shape[0]) + " rows and " + str(df_selected_sector.shape[1]) + "columns")
st.dataframe(df_selected_sector)

#group_sector = dataframe.groupby('GICS Sector')
# print(group_sector.first())
# print(group_sector.describe())
# print(group_sector.get_group("Health Care"))
# print(dataframe.Symbol)

def filedownload(dnl):
    csv = dnl.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv; base64,{b64}" download="SP500.csv"> Download File </a>'
    return href

st.markdown(filedownload(df_selected_sector), unsafe_allow_html=True)


stock_data = yf.download (             #pdr.get_data_yahoo(...
    tickers = list(df_selected_sector[:10].Symbol),
    period = "ytd",
    interval = "1d",
    group_by = "ticker",
    auto_adjust = True,
    prepost = True,
    threads = True,
    proxy = None
)

#print(data['ABT'])
# stock_df = pd.DataFrame(data["ABT"].Close)
# stock_df["Date"] = stock_df.index
# print(stock_df)

def price_plot(symbol):
    stock_df = pd.DataFrame(stock_data[symbol].Close)
    stock_df["Date"] = stock_df.index
    plt.fill_between(stock_df.Date, stock_df.Close, color="skyblue", alpha=0.3)
    plt.plot(stock_df.Date, stock_df.Close, color="skyblue", alpha=0.8)
    plt.xticks(rotation=90)
    plt.title(symbol, fontweight="bold")
    plt.xlabel("Date", fontweight="bold")
    plt.ylabel("Closing Price", fontweight="bold")
    #return plt.show()
    return st.pyplot()

#price_plot("ABT")

num_company = st.sidebar.slider("Number of Companies", 1, 5)
st.set_option('deprecation.showPyplotGlobalUse', False)

if st.button("Show Plot"):
    st.header("Closing Price")
    for i in list(df_selected_sector.Symbol)[:num_company]:
        price_plot(i)


