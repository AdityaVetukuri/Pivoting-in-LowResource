import numpy as np
import string
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Embedding, LSTM, RepeatVector, Dense, TimeDistributed
import matplotlib.pyplot as plt


def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    return model


def read_translation_data(source, target):
    train_data = []
    with open(source) as inp_source:
        source_lines = inp_source.readlines()
    with open(target) as inp_target:
        target_lines = inp_target.readlines()

    for source, target in zip(source_lines, target_lines):
        train_data.append([source, target])
    return train_data

def max_length(lines):
    return max(len(line.split()) for line in lines)


def text_cleaning(data, max_length):
    data[:, 0] = [s.translate(str.maketrans('', '', string.punctuation)) for s in data[:, 0]]
    data[:, 1] = [s.translate(str.maketrans('', '', string.punctuation)) for s in data[:, 1]]
    train_data = []
    for i in range(len(data)):
        if (len(data[i, 0].lower().split()) < max_length):
            train_data.append([data[i, 0].lower().split(), data[i, 1].lower().split()])

    return train_data


def max_length(lines):
    return max(len(line.split()) for line in lines)


def tokenization(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    seq = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    seq = pad_sequences(seq, maxlen=length, padding='post')
    return seq


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_sentence(i):
    temp = []
    for j in range(len(i)):
        t = get_word(i[j], target_tokenizer)
        if j > 0:
            if (t == get_word(i[j - 1], target_tokenizer)) or (t == None):
                temp.append('')
            else:
                temp.append(t)
        else:
            if (t == None):
                temp.append('')
            else:
                temp.append(t)
    return ' '.join(temp)


def get_word(n, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == n:
            return word
    return None


FILENAME_source = './az-en/az-en-tanzil/Tanzil.az-en.az'
FILENAME_target = './az-en/az-en-tanzil/Tanzil.az-en.en'

train_data = np.array(read_translation_data(FILENAME_source, FILENAME_target))

# sampling 50,000 lines
train_data = train_data[:200000, :]

# To know sentence max lengths.
source_lang = []
target_lang = []


max_possible_length = 30
train_data = text_cleaning(train_data, max_possible_length)
train_data = np.array(train_data)
# Tokenizing the data
source_tokenizer = tokenization(train_data[:, 0])
source_vocab_size = len(source_tokenizer.word_index) + 1
source_length = 30

target_tokenizer = tokenization(train_data[:, 1])
target_length = 50
target_vocab_size = len(target_tokenizer.word_index) + 1

print(source_vocab_size)
print(target_vocab_size)

# populate the lists with sentence lengths
for i in train_data[:, 0]:
    source_lang.append(len(i))

for i in train_data[:, 1]:
    target_lang.append(len(i))

length_df = pd.DataFrame({'Source': source_lang, 'Target': target_lang})

length_df.hist(bins=30)
plt.show()

# split data into train and test set
train, test = train_test_split(train_data, test_size=0.2, random_state=12)

# prepare training data
trainX = encode_sequences(source_tokenizer, source_length, train[:, 0])
trainY = encode_sequences(target_tokenizer, target_length, train[:, 1])

# prepare validation data
testX = encode_sequences(source_tokenizer, source_length, test[:, 0])
testY = encode_sequences(target_tokenizer, target_length, test[:, 1])

# buiilding the model
#
model = define_model(source_vocab_size,target_vocab_size,source_length,target_length,256)

#optimizer
optimizer = optimizers.Adam(lr=0.001)
model.compile(optimizer= optimizer, loss='sparse_categorical_crossentropy')

filename = 'local_repo'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=False, mode='min')

# train model
history = model.fit(trainX,  trainY.reshape(trainY.shape[0], trainY.shape[1], 1), epochs=50, batch_size=32, validation_split= 0.2, callbacks=[checkpoint], verbose=1)

# plotting validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train','validation'])
plt.show()
