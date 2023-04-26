# Checkpoint Objective
#  Build a Streamlit app that predicts the type of iris flower based on user input using a Random Forest Classifier.

# Instructions
# 1. Import the necessary libraries: Streamlit, sklearn.datasets, and sklearn.ensemble.
# 2. Load the iris dataset using the "datasets.load_iris()" function and assign the data and target variables to "X" and "Y", respectively.
# 3. Set up a Random Forest Classifier and fit the model using the "RandomForestClassifier()" and "fit()" functions.
# 4. Create a Streamlit app using the "streamlit.title()" and "streamlit.header()" functions to add a title and header to the app.
# 5. Add input fields for sepal length, sepal width, petal length, and petal width using the "streamlit.slider()" function. Use the minimum, maximum, and mean values of each feature as the arguments for the function.
# 6. Define a prediction button using the "streamlit.button()" function that takes in the input values and uses the classifier to predict the type of iris flower.
# 7. Use the "streamlit.write()" function to display the predicted type of iris flower on the app.
# 8. Deploy your streamlit app with streamlit share
# Note: Make sure to run the app using the "streamlit run" command in your terminal.





import streamlit as st 
# from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')



# import some data to play with
data = pd.read_csv(r'C:\Users\HP\OneDrive\Documents\Data_Science\Data_set\iris.data.csv', header = None)

data.rename(columns = {0: 'sepal length (cm)', 1: 'sepal width (cm)', 2: 'petal length (cm)', 3:  'petal width (cm)', 4: 'names'}, inplace = True)

x = data.drop(['names'], axis = 1)
y = data['names']

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split data into train and test
x_train , x_test , y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify = y)

# create dataframe for train data and test data.
train_data = pd.concat([x_train, pd.Series(y_train)], axis = 1)
test_data = pd.concat([x_test, pd.Series(y_test)], axis = 1)

# Model Creation
logReg = LogisticRegression()
logRegFitted = logReg.fit(x_train, y_train)
y_pred = logRegFitted.predict(x_test)

# using R2 score module from sklearn metrics for the goodness to fit information
score = r2_score(y_test,y_pred)
print(f'the score is {score} Good Job')
print(data.head())

# saving the model using joblib
import joblib
joblib.dump(logReg, 'Logistic_Model.pkl')

# -----------------------------------------------------
# FROM HERE WE BEGIN THE IMPLEMENTATION FOR STREAMLIT.

st.header('IRIS MODEL DEPLOYMENT')
user_name = st.text_input('Register User')

if(st.button('SUBMIT')):
    st.text(f"You are welcome {user_name}. Enjoy your usage")

st.write(data)

from PIL import Image
image = Image.open(r'image_entry.jpg')
st.sidebar.image(image)


st.sidebar.subheader(f"Hey {user_name}")
metric = st.sidebar.radio('How do you want your feature input?\n \n \n', ('slider', 'direct input'))


if metric == 'slider':
   sepal_length = st.sidebar.slider('SEPAL LENGTH', 0.0, 9.0, (5.0))

   sepal_width = st.sidebar.slider('SEPAL WIDTH', 0.0, 4.5, (2.5))

   petal_length = st.sidebar.slider('PETAL LENGTH', 0.0, 8.0, (4.5))

   petal_width = st.sidebar.slider('PETAL WIDTH', 0.0, 3.0, (1.5))
else:
    sepal_length = st.sidebar.number_input('SEPAL LENGTH')
    sepal_width = st.sidebar.number_input('SEPAL WIDTH')
    petal_length = st.sidebar.number_input('PETAL LENGTH')
    petal_width = st.sidebar.number_input('PETAL WIDTH')

st.write('selected inputs: ', [sepal_length, sepal_width, petal_length, petal_width])

input_values = [[sepal_length, sepal_width, petal_length, petal_width]]


# Modelling
# import the model
model = joblib.load(open('Logistic_Model.pkl', 'rb'))
pred = model.predict(input_values)


# fig, ax = plt.subplots()
# ax.scatter(y_pred, y_test)
# st.pyplot(fig)


if pred == 0:
    st.success('The Flower is an  ')
    setosa = Image.open('Iris-setosa.jpg')
    st.image(setosa, caption = 'Iris-setosa', width = 400)
elif pred == 1:
    st.success('The Flower is an Iris-versicolor ')
    versicolor = Image.open('Iris-versicolor.jpg')
    st.image(versicolor, caption = 'Iris-versicolor', width = 400)
else:
    st.success('The Flower is an Iris-virginica ')
    virginica = Image.open('Iris-virginica.jpg')
    st.image(virginica, caption = 'Iris-virginica', width = 400 )