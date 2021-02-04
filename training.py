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


for intent in data["intents"]:


    for pattern in intent["patterns"]:

        words_list = nltk.word_tokenize(pattern)

        words.append(words_list)

        documents.append((words_list, intent["tag"]))

        if intent["tag"] not in classes:

            classes.append(intent["tag"])
