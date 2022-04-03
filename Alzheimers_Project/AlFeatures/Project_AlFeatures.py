import streamlit as st
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from sklearn.metrics import confusion_matrix

import matplotlib
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import os
import plotly.express as px

csv_file = os.path.join(os.path.dirname(__file__), 'alzheimer.csv')
data=pd.read_csv(csv_file)
data["SES"].fillna(data["SES"].mean(), inplace=True)
data["MMSE"].fillna(data["MMSE"].mean(), inplace=True)
data = data[data["Group"]!= "Converted"]
data['Group'].replace(['Nondemented', 'Demented'],[0, 1], inplace=True)
data['M/F'].replace(['M', 'F'],[0, 1], inplace=True)

def createModel():
    model = keras.Sequential([
    keras.Input(shape=(9,)),
    keras.layers.Dense(128, activation='relu' ),
    keras.layers.Dense(32, activation='relu' ),
    keras.layers.Dense(8, activation='relu' ),
    keras.layers.Dense(2, activation='softmax')
    ])
    h5_file = os.path.join(os.path.dirname(__file__), 'bi_fnn.h5')
    model.load_weights(h5_file)
    return model

st.header("Classification of Demented/NonDemented using Alzheimer feature dataset")
st.subheader("Predicts the diagnosis of Alzheimer's disease based on the subjects personal data")
st.write("This application uses Fully Connected Neural network")

#features = st.beta_container()
predictor = st.container()    

with predictor:
    sel = st.selectbox('Pick the gender of the subject', ['Male','Female'])
    M_F = 0
    if sel == 'Female':
        M_F = 1
    Age = st.slider('Age of the subject ', 60, 98,step = 1)
    Age = (Age - data['Age'].min())/(data['Age'].max() - data['Age'].min()) # normalizing

    EDUC = st.slider('Years of Education of the subject ', 6, 23,step = 1)
    EDUC = (EDUC - data['EDUC'].min())/(data['EDUC'].max() - data['EDUC'].min()) # normalizing

    SES = st.slider('Socioeconomic Status of the subject ', 1, 5,step = 1)
    SES = (SES - data['SES'].min())/(data['SES'].max() - data['SES'].min()) # normalizing

    MMSE = st.slider('Mini Mental State Examination of the subject ', 4, 30,step = 1)
    MMSE = (MMSE - data['MMSE'].min())/(data['MMSE'].max() - data['MMSE'].min()) # normalizing
   
    CDR = st.slider('Clinical Dementia Rating of the subject ', 0, 3,step = 1)
    CDR = (CDR - data['CDR'].min())/(data['CDR'].max() - data['CDR'].min()) # normalizing

    eTIV = st.slider('Estimated total intracranial volume of the subject ', 1106, 2004)
    eTIV = (eTIV - data['eTIV'].min())/(data['eTIV'].max() - data['eTIV'].min()) # normalizing   

    nWBV = st.slider('Normalize Whole Brain Volume of the subject ', 0.644, 0.837)
    nWBV = (nWBV - data['nWBV'].min())/(data['nWBV'].max() - data['nWBV'].min()) # normalizing

    ASF = st.slider('Atlas Scaling Factor of the subject ', 0.876, 1.587)
    ASF = (ASF - data['ASF'].min())/(data['ASF'].max() - data['ASF'].min()) # normalizing

    xTest = np.array([M_F , Age , EDUC, SES, MMSE, CDR, eTIV, nWBV ,ASF])
    #st.text(xTest)
    xTest =np.expand_dims(xTest, axis=0)
    #xTest = tf.convert_to_tensor(xTest)
    model = createModel()
    y_pred=model.predict(xTest)
    #st.text(y_pred)
    class_names = ['Nondemented', 'Demented']
    string = "The subject is predicted to be: " + class_names[np.argmax(y_pred)]
    st.success(string)