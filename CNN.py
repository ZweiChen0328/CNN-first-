import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
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
with open("all.csv", "r") as f:
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
print('初始資料顯示')
print(data.head())
print(data.shape)
print(data.info())
print(data['label'].value_counts())

print('資料轉換(數值化、刪除)')
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

# print(activities)
# for activity in activities:
#     data_for_plot = data[(data['label'] == activity)][:Fs*10]
#     plot_activity(activity, data_for_plot)

df = data.drop(['total','time'], axis = 1).copy()
print(df.head())
print(df['label'].value_counts())

print('標籤數值化')
walk = df[df['label']=='walk'].copy()#walk = df[df['label']=='walk'].head(2000).copy()
jump = df[df['label']=='jump'].copy()
train = df[df['label']=='train'].copy()
balanced_data = pd.DataFrame()
balanced_data = balanced_data.append([walk, jump, train])
print('list')
label = LabelEncoder()
balanced_data['label'] = label.fit_transform(balanced_data['label'])
print(balanced_data.head())
print(label.classes_)

print('常態分布化，平均值會變為0, 標準差變為1')
X = balanced_data[['x', 'y', 'z']]
y = balanced_data['label']
scaler = StandardScaler()
X = scaler.fit_transform(X)
scaled_X = pd.DataFrame(data = X, columns = ['x', 'y', 'z'])
scaled_X['label'] = y.values
print(scaled_X.head())

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

print('資料分區')
X, y = get_frames(scaled_X, frame_size, hop_size)#資料總數/兩秒的Hz
print(X.shape, y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0, stratify = y)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)# X_train = X_train.reshape(228, 400, 3, 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)# X_test = X_test.reshape(58, 400, 3, 1)

print(X_train.shape)

print('CNN_MaxPooling:')
model = Sequential()
model.add(Conv2D(16, (2, 2), activation = 'relu', input_shape = X_train[0].shape,padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# model.add(Dropout(0.2))

model.add(Conv2D(32, (2, 2), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# model.add(Dropout(0.35))

model.add(Flatten())

model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(3, activation='softmax'))

def plot_learningCurve(history, epochs):
      # Plot training & validation accuracy values
  epoch_range = range(1, epochs+1)
  plt.plot(epoch_range, history.history['accuracy'])
  plt.plot(epoch_range, history.history['val_accuracy'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(epoch_range, history.history['loss'])
  plt.plot(epoch_range, history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()


epoch = 50 #訓練&呈現
model.compile(optimizer=Adam(learning_rate = 0.0001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(X_train, y_train, batch_size = 16, epochs = epoch, validation_data= (X_test, y_test), verbose=1)
plot_learningCurve(history, epoch)
y_pred = model.predict_classes(X_test)
mat = confusion_matrix(y_test, y_pred)
print(mat)
plot_confusion_matrix(conf_mat = mat, class_names = label.classes_, show_normed=True, figsize=(7,7))
plt.show()
scores=model.evaluate(X_test, y_test)
print("MaxPooling Accuracy=", scores[1])
# Saving model
model.save('max_model.h5')




print('CNN_AveragePooling:')
model2 = Sequential()

model2.add(Conv2D(16, (2, 2), activation = 'relu', input_shape = X_train[0].shape,padding='same'))
model2.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model2.add(Dropout(0.2))

model2.add(Conv2D(32, (2, 2), activation='relu',padding='same'))
model2.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model2.add(Dropout(0.35))

model2.add(Flatten())

model2.add(Dense(64, activation = 'relu'))
model2.add(Dropout(0.5))

model2.add(Dense(3, activation='softmax'))


model2.compile(optimizer=Adam(learning_rate = 0.0001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
history = model2.fit(X_train, y_train, batch_size = 10, epochs = epoch, validation_data= (X_test, y_test), verbose=1)
plot_learningCurve(history, epoch)

y_pred = model2.predict_classes(X_test)
mat = confusion_matrix(y_test, y_pred)
print(mat)
plot_confusion_matrix(conf_mat = mat, class_names = label.classes_, show_normed=True, figsize=(7,7))
plt.show()
scores=model2.evaluate(X_test, y_test)
print("AveragePooling Accuracy=", scores[1])
# Saving model
model2.save('ave_model.h5')