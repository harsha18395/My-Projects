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
label_to_class = {
    'Nondemented': 0,
    'Demented': 1
}

def scale(df):
    dfs = df.copy()
    for feature in df.columns:
        if feature == 'Group' or feature == 'M/F':
            continue
        max_val = dfs[feature].max()
        min_val = dfs[feature].min()
        scaled = (dfs[feature] - min_val)/(max_val - min_val)
        dfs[feature] = scaled
    return dfs 

def createModel():
    model = keras.Sequential([
    keras.Input(shape=(9,)),
    keras.layers.Dense(128, activation='relu' ),
    keras.layers.Dense(32, activation='relu' ),
    keras.layers.Dense(8, activation='relu' ),
    keras.layers.Dense(2, activation='softmax')
    ])
    return model

header = st.container()
dataset = st.container()
#features = st.beta_container()
model_training = st.container()

with header:
    st.title('Classification of Demented/NonDemented using Alzheimer feature dataset')

with dataset:
    st.header('Alzheimer dataset overview')
    st.text('The dataset was downloaded from Kaggle - Alzheimer features :')
    st.write(data.head())
    st.text('Features explanation:')
    st.text('Group --> Class\nAge --> Age\nEDUC --> Years of Education\nSES --> Socioeconomic Status / 1-5\nMMSE --> Mini Mental State Examination\n\
CDR --> Clinical Dementia Rating\neTIV --> Estimated total intracranial volume\nnWBV --> Normalize Whole Brain Volume\nASF --> Atlas Scaling Factor')
    st.text('Value counts' )
    st.write(data["Group"].value_counts())
    fig = px.pie(data,names = "Group",title='Classification of Dataset')
    st.write(fig)
    data['Group'].replace(['Nondemented', 'Demented'],[0, 1], inplace=True)
    data['M/F'].replace(['M', 'F'],[0, 1], inplace=True)
    corr = data.corr()
    st.text('Correlation table')
    fig = plt.figure(figsize=(14,8))
    sns.heatmap(corr, 
        cmap="Blues", annot=True,
        xticklabels=corr.columns,
        yticklabels=corr.columns)
    st.pyplot(fig)   

with model_training:
    st.header('Lets train and test dataset')
    scaled_data = scale(data)
    y = scaled_data['Group']
    x = scaled_data.drop('Group',axis=1)
    data = data.drop('Group',axis=1)
    xTrain,xTest,yTrain,yTest = train_test_split(x,y,test_size=0.2,random_state=42)
    st.text(f'Number of training data : {len(list(yTrain))}')
    st.text(f'Number of testing data : {len(list(yTest))}')
    yTrain = keras.utils.to_categorical(yTrain,2)
    yTest = keras.utils.to_categorical(yTest,2)
    sel_model = st.selectbox('Pick the AI model to train the data with', ['Fully connected neural network'])
    if sel_model== 'Fully connected neural network':
        model = createModel()
        h5_file = os.path.join(os.path.dirname(__file__), 'bi_fnn.h5')
        model.load_weights(h5_file)
    y_pred=model.predict(xTest)
    y_preds = np.argmax(y_pred, axis=1)
    y_trues = np.argmax(yTest, axis=1)
    st.text('Results for classification using'+ sel_model)
    st.text("accuracy (testing): %.2f%%"% (accuracy_score(y_preds,y_trues)*100.0))
    st.text("F1 Score (testing): %.2f%%"% (f1_score(y_preds,y_trues, average='weighted')*100.0))
    #st.write(classification_report(y_preds,y_trues))
    cm = confusion_matrix(y_trues, y_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'shrink': .3}, linewidths=.1, ax=ax)
    ax.set(
    xticklabels=list(label_to_class.keys()),
    yticklabels=list(label_to_class.keys()),
    title='confusion matrix',
    ylabel='True label',
    xlabel='Predicted label'
    )
    params = dict(rotation=45, ha='center', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), **params)
    plt.setp(ax.get_xticklabels(), **params)
    st.pyplot(fig)