import json

import random

import pickle

import numpy as np

import nltk


from nltk.stem import WordNetLemmatizer





from tensorflow.keras.models import Sequential                       # A sequential model consists of a linear stack of layers.

from tensorflow.keras.layers import Dense,Activation,Dropout

from tensorflow.keras.optimizers import SGD



lemmatizer = WordNetLemmatizer()                     # lemmatizer is an "instance" of the WordNetLemmatizer class.


with open("swiggy_intents.json") as file:

    swiggy_data = json.load(file)





words = []

classes = []

documents = []


ignore_letters = ["?", "!", ",", ":", "."]


for intent in swiggy_data["intents"]:


    for pattern in intent["patterns"]:

        words_list = nltk.word_tokenize(pattern)

        words.extend(words_list)

        documents.append((words_list, intent["tag"]))

        if intent["tag"] not in classes:

            classes.append(intent["tag"])



words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]


words = sorted(list(set(words)))

classes = sorted(list(set(classes)))



with open("data.pickle", "wb") as data_file:

    pickle.dump((words,classes,documents), data_file)




training = []

#output = []


out_empty = [0 for _ in range(len(classes))]


for document in documents:

    bag = []


    word_pattern = document[0]

    word_pattern = [lemmatizer.lemmatize(word.lower()) for word in word_pattern if word not in ignore_letters]



    for word in words:

        if word in word_pattern:

            bag.append(1)

        else:

            bag.append(0)


    output_row = out_empty[::1]

    output_row[classes.index(document[1])] = 1


    training.append([bag, output_row])



random.shuffle(training)              # We are going to shuffle our training data.


training = np.array(training)



training_x = list(training[::1, 0])   # Our "bag of words" lists. (Grab all the elements from the 0th column)

training_y = list(training[::1, 1])   # Our "output rows". (Grab all the elements from the 1st column)


# Using the training_x & training_y "lists", we are going to train our neural network.
# We are going classify each bag_of_words list to an "intent" (class).



model = Sequential()                # Our model will contain a linear stack of layers.


model.add(Dense(128, input_shape = (len(training_x[0]),), activation = "relu"))                  # Our model's input layer.

model.add(Dropout(0.5))

model.add(Dense(64, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(len(training_y[0]), activation = "softmax"))           # Our model's output layer. ("softmax" activated)

sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)

model.compile(loss = "categorical_crossentropy", optimizer = sgd, metrics = ['accuracy'])

hist = model.fit(x= np.array(training_x), y= np.array(training_y), epochs = 200, batch_size = 5, verbose = 1)

model.save("swiggy_model.h5", hist)


print()


print("Done!")
