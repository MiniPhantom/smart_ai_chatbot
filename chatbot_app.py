import random

import json

import pickle

import numpy as np

import nltk

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

from tensorflow.keras.models import load_model






with open("swiggy_intents.json") as file:

    swiggy_data = json.load(file)




with open("data.pickle", "rb") as f:

    words, classes, documents = pickle.load(f)




model = load_model("swiggy_model.h5")


ignore_letters = ["?", "!", ",", ":", "."]



def clean_up_sentence(sentence):


    sentence_words = nltk.word_tokenize(sentence)


    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words if word not in ignore_letters]


    return sentence_words

    # This is how we clean up a sentence provided by the user.



def bag_of_words(sentence):


    sentence_words = clean_up_sentence(sentence)


    bag = [0 for _ in range(len(words))]



    for w in sentence_words:



        for i, word in enumerate(words):



            if w == word:


                bag[i] = 1

            else:

                bag[i] = 0



    return np.array(bag)


    # This is how we create and return our "bag_of_words" list as a numpy array.



def predict_class(sentence):

    b_o_g = bag_of_words(sentence)                   # We get our "bag_of_words" numpy array like this.




    res = model.predict(np.array([b_o_g]))[0]



    ERROR_THRESHOLD = 0.25

    results = [[i, r] for (i, r) in enumerate(res) if r > ERROR_THRESHOLD]



    results.sort(key = lambda x : x[1] , reverse = True)


    return_list = []


    for r in results:


        return_list.append({"intent" : classes[r[0]], "probability" : str(r[1])})



    return return_list


def get_response(result_list, intents_data):


    tag = result_list[0]["intent"]

    list_of_intents = swiggy_data["intents"]

    for intent in list_of_intents:

        if intent["tag"] == tag:

            result = random.choice(intent["responses"])

            break


    return result


def chat():

    print("Swiggy_Bot: How may I help you?      (Type 'quit' to end this chat session)")


    while True:



        inp = input("You: ")

        if(inp.lower() == "quit"):

            print("Swiggy_Bot: I really enjoyed chatting with you. Goodbye!")

            break

        else:

            ints = predict_class(inp)

            response = get_response(ints, swiggy_data)


            print("Swiggy_Bot: ", response)
















if __name__ == "__main__":


    # Do Something


    chat()
