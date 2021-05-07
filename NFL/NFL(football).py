import base64
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title("NFL Football Stats Explorer (based on rushing)")

st.markdown('''
This app provides simple scraping of NFL Football player stats

Data-source : [pro-football-reference.com]

''')

st.sidebar.header("User Input Features")
year_select = st.sidebar.selectbox("Year", list(reversed(range(1990,2020))))

@st.cache
def load_data(year):
    url = "https://www.pro-football-reference.com/years/" + str(year) + "/rushing.htm"
    html = pd.read_html(url, header = 1)
    dataframe = html[0]
    raw_data = dataframe.drop(dataframe[dataframe.Age == "Age"].index)
    raw_data = raw_data.fillna(0)
    playerstats = raw_data.drop(['Rk'], axis = 1)
    return playerstats
playerstats = load_data(year_select)

sorted_unique_team = sorted(playerstats.Tm.unique())
team_select = st.sidebar.multiselect("Team", sorted_unique_team, sorted_unique_team)

unique_post = ["RB", "QB", "WR", "FB", "TE"]
post_select = st.sidebar.multiselect("Position", unique_post, unique_post)

dataframe_select = playerstats[(playerstats.Tm.isin(team_select)) & (playerstats.Pos.isin(unique_post))]

st.header("Display Player stats of elected teams")
st.write("Data Dimension: " + str(dataframe_select.shape[0]) + " rows and " + str(dataframe_select.shape[1]) + "columns" )
st.dataframe(dataframe_select)

def download_file(filename):
    csv_file = filename.to_csv(index=False)
    b64 = base64.b64encode(csv_file.encode()).decode()
    href = f'<a href="data:file/csv_file;base64,{b64}" download="player-stats.csv"> Download File </a>'
    return href

st.markdown(download_file(dataframe_select), unsafe_allow_html=True)

if st.button("Intercorrelation Heatmap"):
    st.header("Intercorrelation Heatmap")
    dataframe_select.to_csv('output.csv', index=False)  #made a csv for heatmap to function
    df = pd.read_csv('output.csv')

    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    st.set_option('deprecation.showPyplotGlobalUse', False)

    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7,5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
    st.pyplot()

