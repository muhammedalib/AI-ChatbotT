# coding: utf8 for turkish letters
import numpy as np
import Functions
from TurkishStemmer import TurkishStemmer
turkStem = TurkishStemmer()
import nltk
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import random
import json

word2vec = {}
Word = []
Vectors = []
with open('vectors.txt', encoding='utf8') as f: #reading pre-trained glove vectors
    for line in f:
        values = line.split()
        w = values[0]
        Word.append(w)    # appending pre-trained glove vectors's words to word list
        vec = np.asarray(values[1:], dtype='float32')
        Vectors.append(vec)    #appending pre-trained glove vectors's vectors to vector list
        word2vec[w] = vec      #matching words to its vector and combining them into one list



glove_input = 'vectors.txt'
word2vec_output = 'vectors.word2vec'
glove2word2vec(glove_input, word2vec_output) #changing glove to word2vec format


my_model = KeyedVectors.load_word2vec_format(word2vec_output, binary=False) #loading model

words = []
classes = []
documents = []
intents = json.loads(open('intents.json', encoding='utf8').read())   #reading intens, tag and responses from intens.json file

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)  # tokenize each word
        words.extend(wrds)
        documents.append((wrds, intent['tag']))
        # add documents in the data

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])




def dataset_generator(data): # create our training data
    def _dataset_generator(self):
        random.shuffle(data)   #shuffling data
        output_empty = [0] * len(classes)  # create a full of zero  array for our output

        for doc in documents:
            clear_pattern_words = []
            time_count = 0 #word counter in one sentence
            pattern_words = doc[0]
            data_x = []
            for a in pattern_words:
                clear_pattern_words = Functions.clear_sentence(a) #cleared version of sentences
            for w in Word:

                if w in pattern_words:   #checking if the not clear word is in our pre-trained data or not
                    data_x.append(my_model[w]) #if is in the data use pre-trained vector
                    time_count = time_count + 1
                    print(':',w)

                elif w not in pattern_words and w in clear_pattern_words: #checking if the  clear word is in our pre-trained data or not
                    data_x.append(my_model[w]) #if is in the data use pre-trained vector
                    time_count = time_count + 1
                    print(':Clearlı',w)

            data_x = list(filter(lambda x: x is not None, data_x))
            data_x = np.array(data_x)   #changing data to numpy array
            if data_x.size == 0 or data_x.size % 300 != 0: #checking if size is wrong or empty
                print(f"error ")
                continue

            data_x= data_x.reshape((1, time_count, 300)) #reshaping our data to 3d
            data_y = list(output_empty)
            data_y[classes.index(doc[1])] = 1  #attaching this word to its tag
            data_y = np.array(data_y).reshape((1, 1, len(classes)))  #reshaping tag data to 3d

            print(data_y)
            yield data_x, data_y  # for each time 1 sentence
    return _dataset_generator

print("Training data created")