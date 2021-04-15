import pandas as pd
import streamlit as st
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title("NBA Player Stats Explorer")

st.markdown("""
This small app performs the webscraping of NBA player stats data
* **python libraries : ** base64, pandas, streamlit """)

st.sidebar.header("User input")
selected_year = st.sidebar.selectbox("Year", list(reversed(range(1950,2021))))

@st.cache
def load_data(year):
    url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html"
    html = pd.read_html(url, header = 0)
    dataframe = html[0]
    raw = dataframe.drop(dataframe[dataframe.Age == "Age"].index)
    raw = raw.fillna(0)
    playerdata = raw.drop(["Rk"], axis = 1)
    return playerdata

playerdata = load_data(selected_year)


unique_team = sorted(playerdata.Tm.unique())
selected_team = st.sidebar.multiselect("Team", unique_team, unique_team)

unique_pos = ["C", "PF", "SF", "PG", "SG"]
selected_pos = st.sidebar.multiselect("Position", unique_pos, unique_pos)

df_selected_team = playerdata[(playerdata.Tm.isin(selected_team)) & (playerdata.Pos.isin(selected_pos))]

st.header("Dislay player stats of selected team(s)")
st.write("Data Dimension is " + str(df_selected_team.shape[0]) + " Rows and " + str(df_selected_team.shape[1]) + " columns")
st.dataframe(df_selected_team)

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()       # bytes conv
    href = f'<a href="data:file/csv;base64, {b64}" download="playerdata.csv"> Download CSV here </a>'
    return href

st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

#Heatmap

if st.button("Intercorrelation Heatmap"):
    st.header("Intercorrelation Matrix Heatmap")
    df_selected_team.to_csv("output.csv", index=False)
    df = pd.read_csv('output.csv')

    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7,5))
        st.set_option('deprecation.showPyplotGlobalUse', False)  #disabling a update warning
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
    st.pyplot()