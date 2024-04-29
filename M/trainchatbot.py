import nltk
nltk.download('punkt')
nltk.download('wordnet')


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random


words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('C:\\Users\\TEJA09\\Desktop\\MINI - 1\\M\\intents.json').read()
intents = json.loads(data_file)



for intent in intents['intents']:
    for pattern in intent['patterns']:

        # take each word and tokenize it
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # adding documents
        documents.append((w, intent['tag']))

        # adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

"""
print (len(documents), "documents")

print (len(classes), "classes", classes)

print (len(words), "unique lemmatized words", words)
"""

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# Initialize training data
training = []
output_empty = [0] * len(classes)

for doc in documents:   #initializing training data
    bag = []   #list of tokenized words for the patterns
    pattern_words = doc[0]  #lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
#create  our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
#output is a '0' for  each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
#shutt
random.shuffle(training)

# Split the training data into input (X) and output (y)
X = np.array([i[0] for i in training])
y = np.array([i[1] for i in training])
print("Training data created ")


model = Sequential()
model.add(Dense(128, input_shape=(len(X[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y[0]), activation='softmax'))


from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
import numpy as np

# Learning rate schedule
def lr_schedule(epoch):
    lr = 0.01
    if epoch > 50:
        lr *= 0.1
    elif epoch > 100:
        lr *= 0.01
    return lr

# Compile the model with SGD optimizer (without decay)
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Define learning rate scheduler callback
lr_scheduler = LearningRateScheduler(lr_schedule)

# Fit the model to the training data with learning rate scheduler callback
hist = model.fit(X, y, epochs=200, batch_size=5, verbose=1, callbacks=[lr_scheduler])

# Save the model to a file
model.save('chatbot_model.h5')

print("Model created")


import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Get predictions on the training data
predictions = model.predict(X)

# Convert predictions to class labels
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y, axis=1)

# Calculate metrics
accuracy = accuracy_score(true_classes, predicted_classes)
precision = precision_score(true_classes, predicted_classes, average='weighted')
recall = recall_score(true_classes, predicted_classes, average='weighted')
f1 = f1_score(true_classes, predicted_classes, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)