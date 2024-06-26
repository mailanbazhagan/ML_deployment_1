import pandas as pd 
import numpy as np 
import seaborn as sns
import pickle 
import streamlit as st 
import matplotlib.pyplot as plt
from PIL import Image 


# loading in the model to predict on the data 
pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in) 

def plot():
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
	column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
	df = pd.read_csv(url, names=column_names)
	# Draw a pairplot to visualize the relationships between variables
	st.subheader('Pairplot of Iris Dataset')
	fig = sns.pairplot(df, hue='class')
	st.pyplot(fig)

    # Draw a violin plot for each species
	st.subheader('Violin Plots of Sepal and Petal Dimensions Grouped by Species')
	fig, axs = plt.subplots(2, 2)
	sns.violinplot(x='class', y='sepal_length', data=df, ax=axs[0, 0])
	sns.violinplot(x='class', y='sepal_width', data=df, ax=axs[0, 1])
	sns.violinplot(x='class', y='petal_length', data=df, ax=axs[1, 0])
	sns.violinplot(x='class', y='petal_width', data=df, ax=axs[1, 1])
	st.pyplot(fig)

def welcome(): 
	return 'welcome all'

# defining the function which will make the prediction using 
# the data which the user inputs 
def prediction(sepal_length, sepal_width, petal_length, petal_width): 

	prediction = classifier.predict( 
		[[sepal_length, sepal_width, petal_length, petal_width]]) 
	print(prediction) 
	return prediction 
	

# this is the main function in which we define our webpage 
def main(): 
	# giving the webpage a title 
	st.title("Iris Flower Prediction") 
	
	# here we define some of the front end elements of the web page like 
	# the font and background color, the padding and the text to be displayed 
	html_temp = """ 
	<div style ="background-color:pink;padding:13px"> 
	<h1 style ="color:black;text-align:center;">Streamlit Iris Flower Classifier ML App </h1> 
	</div> 
	"""
	
	plot()

	# this line allows us to display the front end aspects we have 
	# defined in the above code 
	st.markdown(html_temp, unsafe_allow_html = True) 
	
	# the following lines create text boxes in which the user can enter 
	# the data required to make the prediction 
	sepal_length = st.text_input("Sepal Length") 
	sepal_width = st.text_input("Sepal Width") 
	petal_length = st.text_input("Petal Length") 
	petal_width = st.text_input("Petal Width") 
	result ="" 
	
	# the below line ensures that when the button called 'Predict' is clicked, 
	# the prediction function defined above is called to make the prediction 
	# and store it in the variable result 
	if st.button("Predict"): 
		result = prediction(sepal_length, sepal_width, petal_length, petal_width) 

		if result == 0:
			st.success('The output is Setosa')
			img = Image.open("Setosa.jpeg")
			st.image(img, width=300)
		elif result == 1:
			st.success('The output is Versicolor')
			img = Image.open("Versicolor.jpg")
			st.image(img, width=300)
		else:
			st.success('The output is Virginica')
			mg = Image.open("Virginica.jpeg")
			st.image(img, width=300)

st.write("M Anbazhagan")

	
if __name__=='__main__': 
	main() 
