import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import warnings
import pickle

# https://www.analyticsvidhya.com/blog/2019/06/comprehensive-guide-text-summarization-using-deep-learning-python/

# setup
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")
# nltk.download("stopwords")


# loading data
data = pd.read_csv("../textsummary/Input/Reviews.csv")  # , nrows=100000)  # todo: change this if u want

# basic data cleaning
data.drop_duplicates(subset=['Text'], inplace=True)
data.dropna(axis=0, inplace=True)

# mapping contractions to words
contraction_mapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                       "could've": "could have", "couldn't": "could not",
                       "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                       "hasn't": "has not", "haven't": "have not",
                       "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did",
                       "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                       "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
                       "I'm": "I am", "I've": "I have", "i'd": "i would",
                       "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have", "i'm": "i am",
                       "i've": "i have", "isn't": "is not", "it'd": "it would",
                       "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is",
                       "let's": "let us", "ma'am": "madam",
                       "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                       "mightn't've": "might not have", "must've": "must have",
                       "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
                       "needn't've": "need not have", "o'clock": "of the clock",
                       "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                       "sha'n't": "shall not", "shan't've": "shall not have",
                       "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                       "she'll've": "she will have", "she's": "she is",
                       "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
                       "so've": "so have", "so's": "so as",
                       "this's": "this is", "that'd": "that would", "that'd've": "that would have", "that's": "that is",
                       "there'd": "there would",
                       "there'd've": "there would have", "there's": "there is", "here's": "here is",
                       "they'd": "they would", "they'd've": "they would have",
                       "they'll": "they will", "they'll've": "they will have", "they're": "they are",
                       "they've": "they have", "to've": "to have",
                       "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
                       "we'll've": "we will have", "we're": "we are",
                       "we've": "we have", "weren't": "were not", "what'll": "what will",
                       "what'll've": "what will have", "what're": "what are",
                       "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have",
                       "where'd": "where did", "where's": "where is",
                       "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is",
                       "who've": "who have",
                       "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not",
                       "won't've": "will not have",
                       "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
                       "y'all": "you all",
                       "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are",
                       "y'all've": "you all have",
                       "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                       "you'll've": "you will have", "you're": "you are", "you've": "you have"}

# preprocessing
stop_words = set(stopwords.words('english'))  # stop words from natural language processing library


def text_cleaner(text):
    new_string = text.lower()  # convert all to lowercase
    new_string = BeautifulSoup(new_string, "html.parser").text  # getting rid of all html tag
    new_string = re.sub(r'\([^)]*\)', '', new_string)  # getting rid of punctuation
    new_string = re.sub('"', '', new_string)  # getting rid of quotes
    new_string = ' '.join([contraction_mapping[t]  # replacing all contractions
                           if t in contraction_mapping else t for t in new_string.split(" ")])
    new_string = re.sub(r"'s\b", "", new_string)  # checks for words that end with 's
    new_string = re.sub("[^a-zA-Z]", " ", new_string)  # substitutes all none alphabet with space
    tokens = [w for w in new_string.split() if not w in stop_words]  # creates tokens for everything
    long_words = []
    for i in tokens:
        if len(i) >= 3:  # removing short word
            long_words.append(i)
    return (" ".join(long_words)).strip()


cleaned_text = []
for t in data['Text']:
    cleaned_text.append(text_cleaner(t))


def summary_cleaner(text):
    new_string = re.sub('"', '', text)  # getting rid of quotations
    new_string = ' '.join([contraction_mapping[t]  # replacing all conractions
                           if t in contraction_mapping else t for t in new_string.split(" ")])
    new_string = re.sub(r"'s\b", "", new_string)  # replacing all words that end with 's
    new_string = re.sub("[^a-zA-Z]", " ", new_string)  # replaces all non-alphabet things with space
    new_string = new_string.lower()  # converts to lowercase
    tokens = new_string.split()  # splits into array of characters
    new_string = ''
    for i in tokens:
        if len(i) > 1:
            new_string = new_string + i + ' '  # only adding if words are longer than 1 letter?
    return new_string


# Call the above function
cleaned_summary = []
for t in data['Summary']:
    cleaned_summary.append(summary_cleaner(t))

data['cleaned_text'] = cleaned_text
data['cleaned_summary'] = cleaned_summary
data['cleaned_summary'].replace('', np.nan, inplace=True)  # replaces empty data with NaN
data.dropna(axis=0, inplace=True)  # drops all NaN

data['cleaned_summary'] = data['cleaned_summary'].apply(lambda x: '_START_ ' + x + ' _END_')  # start and end tokens

pickle.dump(data, open("clean_data.pkl", "wb"))

# taking a look at 5 reviews and summaries
# for i in range(5):
#     print("Review:", data['cleaned_text'][i])
#     print("Summary:", data['cleaned_summary'][i])
#     print("\n")


# finding word count
# text_word_count = []
# summary_word_count = []
#
# # populate the lists with sentence lengths
# for i in data['cleaned_text']:
#     text_word_count.append(len(i.split()))
#
# for i in data['cleaned_summary']:
#     summary_word_count.append(len(i.split()))

# length distribution
# length_df = pd.DataFrame({'text': text_word_count, 'summary': summary_word_count})
# length_df.hist(bins=30)
# plt.show()
