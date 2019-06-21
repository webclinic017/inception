# %%
'''Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
'''
from __future__ import print_function

import numpy as np
import pandas as pd
import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

from utils.basic_utils import load_csvs

# %%
print('Loading data...')
profile = load_csvs('summary_detail', ['assetProfile'])

# %%
df = profile.set_index('symbol').loc[:,['longBusinessSummary', 'sector', 'industry']]
df.dropna(subset=['sector', 'industry'], inplace=True)
df.longBusinessSummary.apply(lambda x: len(x)).describe()
summary = df.longBusinessSummary.values
# y, y_labels = pd.factorize(df.industry.values)
y, y_labels = pd.factorize(df.industry.values)

summary_train, summary_test, y_train, y_test = train_test_split(
        summary, y, test_size=0.25, random_state=1000)

max_words = 2500
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(summary_train)
x_train = tokenizer.texts_to_sequences(summary_train)
x_test = tokenizer.texts_to_sequences(summary_test)

# %%
batch_size = 32
epochs = 5

# (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words, 
#                                                          test_split=0.2, )
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

num_classes = np.max(y_train) + 1
print(num_classes, 'classes')

# %%
print('Vectorizing sequence data...')
# tokenizer = Tokenizer(num_words=max_words)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Convert class vector to binary class matrix '
      '(for use with categorical_crossentropy)')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

# %%
print('Building model...')
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# %%
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# model.summary()
# %%
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)

# %%
score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

#%%
i = 0
while i < 50:
    sample = df.sample()
    print(f'{sample.index[0]}, {sample.longBusinessSummary[0][:30]}... >> {sample.industry.values[0]}')
    x_pred = tokenizer.texts_to_sequences(sample.longBusinessSummary.values)
    x_pred = tokenizer.sequences_to_matrix(x_pred, mode='binary')
    pred_class = model.predict_classes(x_pred)
    print(y_labels[pred_class[0]])
    i+=1

#%%
