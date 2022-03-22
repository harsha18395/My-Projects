import streamlit as st
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import copy
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet import ResNet50
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

W = 224
H = 224
#168

label_to_class = {
    'MildDemented': 0,
    'ModerateDemented': 1,
    'NonDemented': 2,
    'VeryMildDemented':3
    }
class_to_label = {v: k for k, v in label_to_class.items()}
n_classes = len(label_to_class)

@st.cache
def get_images(dir_name=os.path.join(os.path.dirname(__file__),'AlzheimerDataset'), label_to_class=label_to_class):
    """read images / labels from directory"""
    Images = []
    Classes = []
    for j in ['/train','/test']:
        for label_name in os.listdir(dir_name+str(j)):
            cls = label_to_class[label_name]
            for img_name in os.listdir('/'.join([dir_name+str(j), label_name])):
                img = load_img('/'.join([dir_name+str(j), label_name, img_name]), target_size=(W, H))
                img = img_to_array(img)
                Images.append(img)
                Classes.append(cls)
            
    Images = np.array(Images, dtype=np.float32)
    Classes = np.array(Classes, dtype=np.float32)
    Images, Classes = shuffle(Images, Classes, random_state=0)
    
    return Images, Classes

Images, Classes = get_images()

def createResnetModel():
    model = keras.models.Sequential()
    # load model
    ResNet = ResNet50(include_top=False, input_shape=(W,H,3),pooling='avg')    
    # Freezing Layers
    for layer in ResNet.layers:
        layer.trainable=False    
    model.add(ResNet)
    model.add(Flatten())
    model.add(Dense(512,activation='relu',kernel_initializer='he_uniform'))
    model.add(Dense(128,activation='relu',kernel_initializer='he_uniform'))
    model.add(Dense(32,activation='relu',kernel_initializer='he_uniform'))
    model.add(Dense(n_classes,activation='softmax'))
    return model

def createVGGModel():
    model = keras.models.Sequential()
    # load model
    VGG = VGG16(include_top=False, input_shape=(W,H,3),pooling='avg')    
    # Freezing Layers
    for layer in VGG.layers:
        layer.trainable=False    
    model.add(VGG)
    model.add(Flatten())
    model.add(Dense(512,activation='relu',kernel_initializer='he_uniform'))
    model.add(Dense(128,activation='relu',kernel_initializer='he_uniform'))
    model.add(Dense(32,activation='relu',kernel_initializer='he_uniform'))
    model.add(Dense(n_classes,activation='softmax'))
    return model

def preprocess(sel_model):
    if sel_model == 'Resnet50':
        from keras.applications.resnet import preprocess_input
    elif  sel_model == 'VGG16':
        from keras.applications.vgg16 import preprocess_input
    datagen_train = ImageDataGenerator(
        preprocessing_function=preprocess_input, # image preprocessing function
        rotation_range=30,                       # randomly rotate images in the range
        width_shift_range=0.1,                   # randomly shift images horizontally
        height_shift_range=0.1,                  # randomly shift images vertically
        horizontal_flip=True,                    # randomly flip images horizontally
        vertical_flip=False,                     # randomly flip images vertically
    )
    datagen_test = ImageDataGenerator(
        preprocessing_function=preprocess_input, # image preprocessing function
    )


def Train_Val_Plot(acc,val_acc,loss,val_loss,auc,val_auc,precision,val_precision):
    
    fig, (ax1, ax2,ax3,ax4) = plt.subplots(1,4, figsize= (20,4))
    fig.suptitle(" MODEL'S METRICS VISUALIZATION ")

    ax1.plot(range(1, len(acc) + 1), acc)
    ax1.plot(range(1, len(val_acc) + 1), val_acc)
    ax1.set_title('History of Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend(['training', 'validation'])


    ax2.plot(range(1, len(loss) + 1), loss)
    ax2.plot(range(1, len(val_loss) + 1), val_loss)
    ax2.set_title('History of Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend(['training', 'validation'])
    
    ax3.plot(range(1, len(auc) + 1), auc)
    ax3.plot(range(1, len(val_auc) + 1), val_auc)
    ax3.set_title('History of AUC')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('AUC')
    ax3.legend(['training', 'validation'])
    
    ax4.plot(range(1, len(precision) + 1), precision)
    ax4.plot(range(1, len(val_precision) + 1), val_precision)
    ax4.set_title('History of Precision')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Precision')
    ax4.legend(['training', 'validation'])
    st.pyplot(fig)



header = st.container()
dataset = st.container()
#features = st.beta_container()
model_training = st.container()
visualization = st.container()

with header:
    st.title('Classification of Alzheimer stages using MRI scan dataset')

with dataset:
    st.header('Alzheimer MRI dataset overview')
    st.text('The dataset was downloaded from Kaggle - Alzheimers Dataset ( 4 class of Images ) : ')
    st.text(class_to_label[0])
    st.text(class_to_label[1])
    st.text(class_to_label[2])
    st.text(class_to_label[3])

    st.text('Plotting some sample images with the label : ')
    n_total_images = Images.shape[0]

    for target_cls in [0,1,2,3]:
        indices = np.where(Classes == target_cls)[0] # get target class indices on Images / Classes
        n_target_cls = indices.shape[0]
        label = class_to_label[target_cls]
        print(label, n_target_cls, n_target_cls*100/n_total_images)

        n_cols = 10 # # of sample plot
        fig, axs = plt.subplots(ncols=n_cols, figsize=(25, 3))

        for i in range(n_cols):

            axs[i].imshow(np.uint8(Images[indices[i]]))
            axs[i].axis('off')
            axs[i].set_title(label)

        st.pyplot(fig)



with model_training:
    st.header('Lets train dataset according user selected model : ')
    indices_train, indices_test = train_test_split(list(range(Images.shape[0])), train_size=0.8, test_size=0.2, shuffle=False)
    x_train = Images[indices_train]
    y_train = Classes[indices_train]
    x_test = Images[indices_test]
    y_test = Classes[indices_test]
    y_train = keras.utils.np_utils.to_categorical(y_train, n_classes)
    y_test = keras.utils.np_utils.to_categorical(y_test, n_classes)
    st.text(f'Number of training data : {len(list(y_train))}')
    st.text(f'Number of testing data : {len(list(y_test))}')
    sel_model = st.selectbox('Pick the CNN model to train the data with', ['Resnet50'])
    preprocess(sel_model)
    if sel_model== 'Resnet50':        
        model = createResnetModel()
        model.load_weights(os.path.join(os.path.dirname(__file__),'resnet.h5'))
        history=np.load(os.path.join(os.path.dirname(__file__),'resnet_history.npy'),allow_pickle='TRUE').item()
    #elif sel_model == 'VGG16':
    #    model = createResnetModel()
    #    model.load_weights(os.path.join(os.path.dirname(__file__),'resnet.h5'))
    #    history=np.load(os.path.join(os.path.dirname(__file__),'my_history.npy'),allow_pickle='TRUE').item()

with visualization:
    st.header('Lets visualize the training metrics for ' + sel_model + ' : ')
    st.text('Metric graphs : ')
    Train_Val_Plot(history['accuracy'],history['val_accuracy'],
               history['loss'],history['val_loss'],
               history['auc'],history['val_auc'],
               history['precision'],history['val_precision']  ) 
    predictions=model.predict(x_test)
    CATEGORIES = ['MildDemented','ModerateDemented','NonDemented','VeryMildDemented']       
    # plot images - prediction actual
    st.text('Metric graphs : ')
    rows = 2
    columns = 5
    fig, axs= plt.subplots(rows, columns , figsize=(25, 10))
    axs = axs.flatten()
    i=10
    for a in axs:
        Image = Images[indices_test[i]]
        pred_label=predictions.argmax(axis=1)[i]
        actual_label=y_test.argmax(axis=1)[i]
        pred_label=CATEGORIES[pred_label]
        actual_label=CATEGORIES[actual_label]
        label= 'pred: '+ pred_label +' '+'real: '+ actual_label
        #axs[i](rows,columns,i)
        a.imshow(np.uint8(Image))
        a.set_title(label)
        i=i+1
    st.pyplot(fig)

    ## plot confusion matrix
    st.text('Confusion Matrix : ')
    y_preds = model.predict(x_test)
    y_preds = np.argmax(y_preds, axis=1)
    y_trues = np.argmax(y_test, axis=1)
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
    st.text("F1 Score (testing): %.2f%%"% (f1_score(y_trues, y_preds, average='weighted')*100.0))
    st.text("accuracy (testing): %.2f%%"% (accuracy_score(y_trues, y_preds)*100.0))
