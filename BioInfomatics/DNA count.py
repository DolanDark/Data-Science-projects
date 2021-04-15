import pandas
import streamlit
import altair
from PIL import Image

dna_image = Image.open("gettyimage.jpg")

streamlit.image(dna_image, use_column_width=True)

streamlit.write("""
# DNA Count Neucleotide

This app counts the neucleotide composition of query DNA

***

""")

streamlit.header("Enter the DNA sequence - ")

seq_input = "> DNA Query\nATCGGCATAAAGCTAGCTGGCGTACGCTATGTCGATCGTCGAT\nCGTATCGATCATCGATGTACATGACGATGCATCTAGCGCATGTA\nCATGCTTCGAAGCTGATAGTGAGCATGTAGCATAGAGCTAATC"

seq = streamlit.text_area("Input Sequence - ", seq_input, height = 250)
seq = seq.splitlines()
seq = seq[1:]

seq = "".join(seq)

streamlit.write('''
***
''')

streamlit.header("Inputed DNA query")
streamlit.write(seq)

streamlit.header("Output DNA Neucleotide count")
streamlit.subheader("1 - Print Dictionary")
def dna_nucleotide_count(sequen):
    d = dict([
                ("A", sequen.count("A")),
                ("T", sequen.count("T")),
                ("G", sequen.count("G")),
                ("C", sequen.count("C"))
            ])
    return d

X = dna_nucleotide_count(seq)
streamlit.write(X)

streamlit.subheader("2 - Print text")
streamlit.write("There are " + str(X["A"]) + " Adenaine (A)")
streamlit.write("There are " + str(X["T"]) + " Thymine (T)")
streamlit.write("There are " + str(X["G"]) + " Guanine (G)")
streamlit.write("There are " + str(X["C"]) + " Cytocinine (G)")


streamlit.subheader("3 - Display Dataframe")
DF = pandas.DataFrame.from_dict(X, orient="index")  #take val from dictionary to plot
DF = DF.rename({0:"count"}, axis="columns")          #renaming the column
DF.reset_index(inplace=True)
DF = DF.rename(columns = {"index": "nucleotide"})
streamlit.write(DF)


streamlit.subheader("4 - Display Bar Chart")
BAR = altair.Chart(DF).mark_bar().encode(x = 'nucleotide', y = 'count')

BAR = BAR.properties(
    width=altair.Step(50)       #controls width of bar
)
streamlit.write(BAR)
