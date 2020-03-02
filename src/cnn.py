from collections import Counter

from imblearn.over_sampling import SMOTE

from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Flatten, Dense, Conv1D, MaxPooling1D
from keras.models import Model, Input


import math
import matplotlib.pyplot as plt

import numpy as np
import os
import pandas as pd
import plotly.express as px
from statistics import mean 
import seaborn as sns

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample


from tqdm import tqdm


complaints_df = pd.read_pickle('../pickle/complaints_v2.pkl')
reports_df = pd.read_pickle('../pickle/reports.pkl')

#add a url feature to match with reports_df
complaints_df['url'] = complaints_df.apply(lambda x: x.complaints['url'],axis=1)
reports_df['url'] = reports_df.apply(lambda x: x.complaints['url'],axis=1)

# Marge, drop duplicates, extract nlp and target, reset indexes
df = reports_df.merge(complaints_df, on='url')
df_t = df.drop_duplicates('url').copy()
df_t = df_t[['complaint_detail','enforcement']]
df_t['enforcement'] = df_t['enforcement'].apply(lambda x: 1 if type(x) == dict else 0 )
df_t.complaint_detail = df_t.complaint_detail.apply(str)
df_t = df_t.drop_duplicates('complaint_detail')
df_t.reset_index(inplace=True)


# init Tokenizer 
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(df_t.complaint_detail)
sequences = tokenizer.texts_to_sequences(df_t.complaint_detail)
word_index = tokenizer.word_index


visualize = False
if visualize:
    # Here we are visualizing the distibution of token lengths
    # this is needed for use to synthisize new data
    lengths=Counter([int(len(i)) for i in sequences])
    plt.figure(figsize=(10,10))
    plt.bar(lengths.keys(),lengths.values(),log=True)
    plt.show()
    pd.DataFrame([int(len(i)) for i in sequences]).describe()


# Resampling
data = pad_sequences(sequences, maxlen=100)
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(data, df_t.enforcement)
labels = to_categorical(np.asarray(y_res))
print('Shape of data tensor:', X_res.shape)
print('Shape of label tensor:', labels.shape)

# Shuffle data, and train test spilt
indices = np.arange(X_res.shape[0])
np.random.shuffle(indices)
X_res = X_res[indices]
labels = labels[indices]
nb_validation_samples = int(.4 * X_res.shape[0])


x_train = X_res[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = X_res[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]


# Downloading pretrained word embeddings
embeddings_index = {}
f = open(os.path.join('../../../../../Data/glove.twitter.27B/', 'glove.twitter.27B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# Create embeddings matrix 
embedding_matrix = np.zeros((len(word_index) + 1, 100))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector




# CNN Model
# Fix embeddings from being trainable
embedding_layer = Embedding(len(word_index) + 1,
                            100,
                            weights=[embedding_matrix],
                            input_length=100,
                            trainable=False)

sequence_input = Input(shape=(100,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(6)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)  # global max pooling
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(labels.shape[1], activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc']
             )

# Fit model
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=10, batch_size=16)
