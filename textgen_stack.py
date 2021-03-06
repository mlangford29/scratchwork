'''
#Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io

from sklearn.ensemble import VotingClassifier
from urllib.request import urlopen


url_list = ['https://www.gutenberg.org/files/1342/1342-0.txt',
'https://www.gutenberg.org/files/84/84-0.txt',
'https://www.gutenberg.org/files/1080/1080-0.txt',
'https://www.gutenberg.org/files/100/100-0.txt',
'https://www.gutenberg.org/ebooks/25525.txt.utf-8',
'https://www.gutenberg.org/ebooks/1635.txt.utf-8',
'https://www.gutenberg.org/files/11/11-0.txt',
'https://www.gutenberg.org/ebooks/844.txt.utf-8',
'https://www.gutenberg.org/ebooks/16328.txt.utf-8',
'https://www.gutenberg.org/files/205/205-0.txt',
'https://www.gutenberg.org/files/2701/2701-0.txt',
'https://www.gutenberg.org/files/98/98-0.txt',
'https://www.gutenberg.org/files/76/76-0.txt',
'https://www.gutenberg.org/files/219/219-0.txt',
'https://www.gutenberg.org/ebooks/5200.txt.utf-8',
'https://www.gutenberg.org/files/1661/1661-0.txt',
'https://www.gutenberg.org/ebooks/174.txt.utf-8',
'https://www.gutenberg.org/files/4300/4300-0.txt',
'https://www.gutenberg.org/files/25344/25344-0.txt',
'https://www.gutenberg.org/files/74/74-0.txt',
'https://www.gutenberg.org/files/2600/2600-0.txt',
'https://www.gutenberg.org/files/1400/1400-0.txt',
'https://www.gutenberg.org/ebooks/1232.txt.utf-8',
'https://www.gutenberg.org/ebooks/3207.txt.utf-8',
'https://www.gutenberg.org/files/28054/28054-0.txt',
'https://www.gutenberg.org/files/3600/3600-0.txt',
'https://www.gutenberg.org/ebooks/19942.txt.utf-8']
text = ''

for book_url in url_list:
    text += urlopen(book_url).read().decode('utf-8').lower()

'''
path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
'''


print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 3
step = 1
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Building model!')

'''
def create_model1():
    model1 = Sequential()
    model1.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model1.add(Dense(len(chars), activation='softmax'))
    optimizer1 = RMSprop(lr=0.01)
    model1.compile(loss='categorical_crossentropy', optimizer=optimizer1)
    return model1

def create_model2():
    model2 = Sequential()
    model2.add(LSTM(256, input_shape=(maxlen, len(chars)), return_sequences=True))
    model2.add(LSTM(256))
    model2.add(Dense(len(chars), activation='softmax'))
    optimizer2 = RMSprop(lr=0.001)
    model2.compile(loss='categorical_crossentropy', optimizer=optimizer2)
    return model2

def create_model3():
    model3 = Sequential()
    model3.add(LSTM(512, input_shape=(maxlen, len(chars)), return_sequences=True))
    model3.add(LSTM(512))
    model3.add(Dense(len(chars), activation='softmax'))
    optimizer3 = RMSprop(lr=0.001)
    model3.compile(loss='categorical_crossentropy', optimizer=optimizer3)
    return model3

model1 = KerasClassifier(builf_fn=create_model1)
model2 = KerasClassifier(builf_fn=create_model2)
model3 = KerasClassifier(builf_fn=create_model3)



model1 = Sequential()
model1.add(LSTM(128, input_shape=(maxlen, len(chars))))
model1.add(Dense(len(chars), activation='softmax'))
optimizer1 = RMSprop(lr=0.01)
model1.compile(loss='categorical_crossentropy', optimizer=optimizer1)


model2 = Sequential()
model2.add(LSTM(256, input_shape=(maxlen, len(chars)), return_sequences=True))
model2.add(LSTM(256))
model2.add(Dense(len(chars), activation='softmax'))
optimizer2 = RMSprop(lr=0.001)
model2.compile(loss='categorical_crossentropy', optimizer=optimizer2)

model3 = Sequential()
model3.add(LSTM(512, input_shape=(maxlen, len(chars)), return_sequences=True))
model3.add(LSTM(512))
model3.add(Dense(len(chars), activation='softmax'))
optimizer3 = RMSprop(lr=0.001)
model3.compile(loss='categorical_crossentropy', optimizer=optimizer3)
'''

'''
model = Sequential()
model.add(LSTM(512, input_shape=(maxlen, len(chars)), return_sequences=True))
model.add(LSTM(512, return_sequences=True))
model.add(LSTM(512))
model.add(Dense(len(chars), activation='softmax'))
optimizer = RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
'''

model = Sequential()
model.add(LSTM(16, input_shape=(maxlen, len(chars)), return_sequences=True))
model.add(LSTM(16, return_sequences=True))
model.add(LSTM(16, return_sequences=True))
model.add(LSTM(16, return_sequences=True))
model.add(LSTM(16, return_sequences=True))
model.add(LSTM(16))
model.add(Dense(len(chars), activation='softmax'))
optimizer = RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.5, 1.0]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y, batch_size=2048, epochs=50, callbacks=[print_callback])

print('Saving model!')
model.save('deep_thin_maxlen2_031120.h5')

