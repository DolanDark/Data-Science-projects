import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

years = [2015, 2016, 2017, 2018, 2019]
str = "https://www.basketball-reference.com/leagues/NBA_{}_per_game.html"

url = str.format(years[0])

#for i in years:
#   url = str.format(i)
#  print(url)

dataframe = pd.read_html(url, header = 0)

#print(dataframe)
#print(len(dataframe))

data_fin = dataframe[0]

#print(data_fin[data_fin.Age == "Age"])

df = data_fin.drop(data_fin[data_fin.Age == "Age"].index)

print(df)

#print(df.shape)

sns.distplot(df.PTS, kde=False)  #for pts
plt.show()

sns.distplot(df.PTS, kde = False, hist_kws= dict(edgecolor="black", linewidth=2), color="#00BFC4")
plt.show()
