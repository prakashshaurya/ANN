import nltk
import pickle
import json 
import numpy as np
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
 
words=[]                         # 
classes =[]                             #
documents=[]
ignored_patterns = ['?' ,'!']

# first read files in string format
data = open('F:/VSCODE/Chatbot-using-Python-master/intents.json').read()
# convert srting to json , whichis dictionary in python
intents = json.loads(data)
 
for intent in intents ["intents"] : 
    for pattern in intent ['patterns'] :
        pattern_words = nltk.word_tokenize(pattern)
        words.extend(pattern_words)

print(words)        




    