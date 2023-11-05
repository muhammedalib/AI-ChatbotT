import random
import re
import nltk
import numpy as np
from tensorflow.python.keras.models import load_model
import Training

f2 = open('stopwords.txt', 'r', encoding='utf8') #stopwords which will be removed from data.
text2 = f2.read()
stop_words = text2.split('\n')
#punctuations= '''!()-[]{};:'"\, <>./?@#$%^&*_~''' # punctuations also will be removed.


model = load_model('chatbot_model.model') #loading trained model

def remove_stop_words(corpus):  #removing stopwords function
    results = []
    tmp = corpus.split(' ')
    for stop_word in stop_words:
        if stop_word in tmp:
            tmp.remove(stop_word)
    results.append(" ".join(tmp))

    return results


def clear_sentence(sentence):   #Clearing sentence function
    results = []
    output_stems = []
    wds = remove_stop_words(sentence)
    input_text = ''.join(wds).lower()   #lowering letters and removing  space
    input_text = re.sub(r'[,.:@#?!&$]', ' ', input_text) #removing punctuations
    results.append(nltk.word_tokenize(input_text)) #tokenizing
    for word in results:

        for w in word:
            if w in Training.Word:
                output_stems.append(w)
            else:
                output_stems.append(Training.turkStem.stem(w)) #stemming (finding root)

    return output_stems


def predict_sentence(text):  # predict sentence function
    temp = clear_sentence(text)  # first clear the incoming sentence
    data_x = []
    for word in temp:
        if word in Training.Word:   #check if the cleared word is in our trained glove model or not
            data_x.append(Training.my_model[word])
        else:
            print(': is not exist on database!', word)

    if len(data_x) == 0 :     # for only one word input which the word  is not in our database
        tag = 'anlamsız'
        return tag
    ERROR_THRESHOLD = 0.20
    data_x = list(filter(lambda x: x is not None, data_x)) # making data a list
    data_x = np.array(data_x).reshape((1, len(data_x), 300)) #shaping our x data to 3d
    result = model.predict(data_x).tolist()[0][-1] # predict
    if max(result) < ERROR_THRESHOLD :
        tag = 'anlamsız'
        return tag
    result_index = np.argmax(result)    #matching word with the class tag
    tag = Training.classes[result_index]
    print(':',max(result))
    print('::',result_index)
    print(max(result))
    print(tag)
    return tag


def getResponse(ints, intents_json): # chatbot response function
    result = []
    tag = ints
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):      # if the input tag matches one of our tag, chatbot  responses randomly
            result = random.choice(i['responses'])
            break
    return result



def chatbot_response(text): # combining functions
    ints = predict_sentence(text)
    res = getResponse(ints, Training.intents)
    return res