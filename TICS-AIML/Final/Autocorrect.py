"""
Eric Zhou
Ms. Lewellen
TICS:AIML
March 9th, 2022

Code is mostly copied from: https://www.geeksforgeeks.org/correcting-words-using-nltk-in-python/

This was my attempt at using NLTK to do some post-processing auto-correction on the words generated
"""

# Imports
from nltk.metrics.distance import edit_distance
from nltk.corpus import words

# NLTK correct words
correct_words = words.words()

# Incorrect spellings
incorrect_words = ['helln', 'wnklw']

for word in incorrect_words:
    temp = [(edit_distance(word, w), w) for w in correct_words if w[0] == word[0]]
    print(sorted(temp, key=lambda val: val[0])[0][1])
