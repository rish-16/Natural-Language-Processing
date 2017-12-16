from pprint import pprint
import numpy as np
import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, LSTM, Embedding
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from keras.preprocessing.text import Tokenizer

data = open('data.txt', 'r').read()

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
encoded = tokenizer.texts_to_sequences([data])[0]

vocab_size = len(tokenizer.word_index) + 1
sequences = list()

for i in range(1, len(encoded)):
	sequence = encoded[i-1:i+1]
	sequences.append(sequence)

X = []
y = []

for j in sequences:
	X.append(j[0])
	y.append(j[1])

X = np.array(X).astype('float32')
y = np.array(y).astype('float32')
y = np_utils.to_categorical(y, vocab_size)

embedding_size = 10

def build_model(vocab_size):
	model = Sequential()
	model.add(Embedding(vocab_size, embedding_size, input_length=1))
	model.add(LSTM(50))
	model.add(Dense(vocab_size, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()

	return model

model = build_model(vocab_size)
model.fit(X, y, epochs=500, verbose=1)

seed = input('Enter seed: ')
if (len(seed.strip()) == 1):
	encoded = tokenizer.texts_to_sequences([seed])[0]
	encoded = np.array(encoded)
	y_pred = model.predict(encoded, verbose=0)
	print (y_pred.all())

	for word, index in tokenizer.word_index.items():
		if (index == y_pred):
			print (word)
else:
	print ('Enter one word only')
