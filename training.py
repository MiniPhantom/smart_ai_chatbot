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

output = []


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

    
