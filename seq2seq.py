import numpy as np
import string
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Embedding, LSTM, RepeatVector, Dense, TimeDistributed
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model
from torchtext.data import bleu_score


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
    return max(len(line) for line in lines)


def text_cleaning(data):
    data[:, 0] = [s.translate(str.maketrans('', '', string.punctuation)) for s in data[:, 0]]
    data[:, 1] = [s.translate(str.maketrans('', '', string.punctuation)) for s in data[:, 1]]
    for i in range(len(data)):
        data[i, 0] = data[i, 0].lower()
        data[i, 1] = data[i, 1].lower()
    return data


def max_length(lines):
    return max(len(line) for line in lines)


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


FILENAME_source = './az-tr-tanzil/Tanzil.az-tr.az'
FILENAME_target = './az-tr-tanzil/Tanzil.az-tr.tr'

train_data = np.array(read_translation_data(FILENAME_source, FILENAME_target))



#HYPER PARAMATERS
training_samples = 200000
optimizer = optimizers.Adam(lr=0.001)
batch_size = 32
epochs = 30
EMBEDDING_DIM = 256
# sampling
train_data = train_data[:training_samples, :]

# To know sentence max lengths.
source_lang = []
target_lang = []

train_data = text_cleaning(train_data)
train_data = np.array(train_data)
# Tokenizing the data
source_tokenizer = tokenization(train_data[:, 0])
source_vocab_size = len(source_tokenizer.word_index) + 1
source_length = max_length(train_data[:,0])

target_tokenizer = tokenization(train_data[:, 1])
target_length = max_length(train_data[:,1])
target_vocab_size = len(target_tokenizer.word_index) + 1

print(source_vocab_size)
print(target_vocab_size)

# populate the lists with sentence lengths
for i in train_data[:, 0]:
    source_lang.append(len(i))

for i in train_data[:, 1]:
    target_lang.append(len(i))

length_df = pd.DataFrame({'Source ': source_lang, 'Target': target_lang})

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
# #
model = define_model(source_vocab_size,target_vocab_size,source_length,target_length, EMBEDDING_DIM)

#optimizer

filename = 'trained_model_az-tr_monday'

model.compile(optimizer= optimizer, loss='sparse_categorical_crossentropy')

checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=False, mode='min')

# train model

with tf.device("/gpu:0"):
    history = model.fit(trainX,  trainY.reshape(trainY.shape[0], trainY.shape[1], 1), epochs= epochs, batch_size= batch_size, validation_split= 0.2, callbacks=[checkpoint], verbose=1)

# plotting validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train','validation'])
plt.show()

model = load_model('trained_model_az-en_50epochs_size_150k_max_length')

preds = model.predict_classes(testX.reshape((testX[:].shape[0], testX.shape[1])))

preds_text = []
for i in preds:
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

    preds_text.append(temp)


targets = [[temp] for temp in test[:,1]]
actuals = [' '.join(temp) for temp in test[:,1]]
predicts = [' '.join(temp) for temp in preds_text]
print(bleu_score(preds_text,targets))
pred_df = pd.DataFrame({'actual': actuals, 'predicted': predicts})
print(pred_df.sample(15))
