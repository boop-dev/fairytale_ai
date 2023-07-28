import random
import torch
import numpy as np
import tensorflow as tf
from nltk.tokenize import RegexpTokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

# read data from dataset
with open('assets/fairytale_dataset.txt', 'r', encoding='utf-8') as dataset:
    data = dataset.read()

data = data[:500000]

tokenizer = RegexpTokenizer(r"\w+")
tokens = tokenizer.tokenize(data.lower())
unique_tokens = sorted(list(set(tokens)))
token_indexes = {token: index for index, token in enumerate(unique_tokens)}
# print(token_indexes)

# setting up context
'''
    context_length: this is the maximum amount of data that can be used by the transformer to predict the next value
    in one instance

    Each chunk of data to be fed into the transformer is context_length + 1 units long.
    Say the chunk is [ 1, 2, 3, 4, 5, 6 ], the first context is [1] and the transformer has to predict the next number
    based on that context. The next context is [ 1, 2 ] and the transformer has to predict the next number based on
    [ 1, 2 ]. This goes on until the context is [ 1, 2, 3, 4, 5 ] (i.e the length of the context only goes up to the
    context_length).
'''
CONTEXT_LENGTH = 10

# Empty arrays to store the contexts and targets
contexts = []
targets = []

# populate the contexts and targets arrays
for t in range(len(tokens) - CONTEXT_LENGTH):
    contexts.append(tokens[t:t + CONTEXT_LENGTH])
    targets.append(tokens[t + CONTEXT_LENGTH])
print(contexts[:100])
print(targets[:100])

# make numpy arrays filled with zeroes for the contexts and targets

contexts_np_array = np.zeros((len(contexts), CONTEXT_LENGTH, len(unique_tokens)), dtype=np.bool_)
targets_np_array = np.zeros((len(contexts), len(unique_tokens)), dtype=np.bool_)
print('finished contexts and targets')

# populate the numpy arrays
for i, words in enumerate(contexts):
    for j, word in enumerate(words):
        contexts_np_array[i, j, token_indexes[word]] = 1
    targets_np_array[i, token_indexes[targets[i]]] = 1

print(targets_np_array.shape)
print('\n\n\n')
print(contexts_np_array.shape)
print('training model')

# Assembling the Neural Network
# model = Sequential()
# model.add(LSTM(128, input_shape=(CONTEXT_LENGTH, len(unique_tokens))))
# model.add(Dense(len(unique_tokens)))
# model.add(Activation('softmax'))
#
# model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01), metrics=["accuracy"])
# model.fit(contexts_np_array, targets_np_array, batch_size=128, epochs=10, shuffle=True)
#
# model.save('assets/fairytaleAiV4-1000000.model')


#
#
model = tf.keras.models.load_model('assets/fairytaleAiV4-1000000.model')
# model.fit(contexts, targets, epochs=200, batch_size=128)
# model.save('assets/fairytaleAiV3-1000000-V2.model')


def predict_next_word(input_text, n_best):
    input_text = input_text.lower()
    X = np.zeros((1, CONTEXT_LENGTH, len(unique_tokens)))
    for i, word in enumerate(input_text.split()):
        X[0, i, token_indexes[word]] = 1

    predictions = model.predict(X)[0]
    return np.argpartition(predictions, -n_best)[-n_best:]


pre = predict_next_word('He wished to marry her because he', 5)
for index in pre:
    print(unique_tokens[index])


def generate_text(input_text, n_words, creativity=3):
    word_sequence = input_text.split()
    current = 0
    for _ in range(n_words):
        sub_sequence = " ".join(tokenizer.tokenize(" ".join(word_sequence).lower())[current:current + n_words])
        try:
            choice = unique_tokens[random.choice(predict_next_word(sub_sequence, creativity))]
        except:
            choice = random.choice(unique_tokens)
        word_sequence.append(choice)
        current += 1
    return " ".join(word_sequence)


print(generate_text("said the king you must not be in such a hurry", 100, 10))

