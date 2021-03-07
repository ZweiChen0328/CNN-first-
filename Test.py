import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
print(tf.version)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import csv
import scipy.stats as stats

from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

processedList = []
tempi = 0
model =  keras.models.load_model('max_model.h5')
model2 =  keras.models.load_model('ave_model.h5')

with open("test.csv", "r") as f:
    for line in f:
        sepline = line.split(",")
        sepline[5] = sepline[5].replace("\n","")
        if tempi == 0:
            tempi = 1
            continue
        temp = [sepline[0], sepline[1], sepline[2], sepline[3],sepline[4],sepline[5]]
        processedList.append(temp)
columns = ['time','x', 'y', 'z','total','label']
data = pd.DataFrame(data = processedList, columns = columns)
data['x'] = data['x'].astype('float')
data['y'] = data['y'].astype('float')
data['z'] = data['z'].astype('float')
Fs = 200
frame_size = Fs*2 # 400
hop_size = Fs # 200
activities = data['label'].value_counts().index
def plot_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 7), sharex=True)
    plot_axis(ax0, data['time'], data['x'], 'X-Axis')
    plot_axis(ax1, data['time'], data['y'], 'Y-Axis')
    plot_axis(ax2, data['time'], data['z'], 'Z-Axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()

def plot_axis(ax, x, y, title):
    ax.plot(x, y, 'g')
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)

df = data.drop(['total','time'], axis = 1).copy()


walk = df[df['label']=='walk'].copy()#walk = df[df['label']=='walk'].head(2000).copy()
jump = df[df['label']=='jump'].copy()
train = df[df['label']=='train'].copy()
balanced_data = pd.DataFrame()
balanced_data = balanced_data.append([walk, jump, train])


label = LabelEncoder()
balanced_data['label'] = label.fit_transform(balanced_data['label'])
X = balanced_data[['x', 'y', 'z']]
y = balanced_data['label']
scaler = StandardScaler()
X = scaler.fit_transform(X)
scaled_X = pd.DataFrame(data = X, columns = ['x', 'y', 'z'])
scaled_X['label'] = y.values

def get_frames(df, frame_size, hop_size):

    N_FEATURES = 3

    frames = []
    labels = []
    for i in range(0, len(df) - frame_size, hop_size):
        x = df['x'].values[i: i + frame_size]
        y = df['y'].values[i: i + frame_size]
        z = df['z'].values[i: i + frame_size]

        # Retrieve the most often used label in this segment
        label = stats.mode(df['label'][i: i + frame_size])[0][0]
        frames.append([x, y, z])
        labels.append(label)

    # Bring the segments into a better shape
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)

    return frames, labels

X, y = get_frames(scaled_X, frame_size, hop_size)
print(X.shape, y.shape)

#資料總數/兩秒的Hz
X1= X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
y_pred = model.predict_classes(X1)
mat = confusion_matrix(y, y_pred)
plot_confusion_matrix(conf_mat = mat, class_names = label.classes_, show_normed=True, figsize=(7,7))
plt.show()
#成功率計算
scores=model.evaluate(X1, y)
print("MaxPooling Accuracy=", scores[1])

X2= X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
y_pred = model2.predict_classes(X2)
mat = confusion_matrix(y, y_pred)
plot_confusion_matrix(conf_mat = mat, class_names = label.classes_, show_normed=True, figsize=(7,7))
plt.show()
scores=model2.evaluate(X2, y)
print("MaxPooling Accuracy=", scores[1])