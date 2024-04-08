
import nltk
import numpy as np
import re
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def quitar_signos(texto):
    texto_sin_signos = re.sub(r'[¿¡]', '', texto)
    return texto_sin_signos

def tokenize(sentence):
    tokens = nltk.word_tokenize(sentence, language='spanish')
    tokens_sin_signos = [quitar_signos(token) for token in tokens]
    return tokens_sin_signos

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag

""" sentence = ['hello', 'how', 'are', 'you']
words = ['hi', 'hello', 'I', 'you', 'bye', 'thank', 'cool']

bag = bag_of_words(sentence, words)
print(bag) """


""" words = ["Organize", "organizes", "organizing"]
stemmed_words = [stem(w) for w in words]
print(stemmed_words) """

""" a = "Hola como estas?"
print(a)

a = tokenize(a)
print(a)

words = [stem(word) for word in a]
print(words) """