from __future__ import unicode_literals, print_function, division
from io import open
import random
import numpy as np
import bcolz
import pickle
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

glove_path = "glove.6B"
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 5000

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def find_pairs(file):
    # Hardcoded per file
    with open(file) as f:

        # Get rid of header file
        _ = f.readline()

        executives = list()
        analysts = list()


        count = 0
        while True:
            # Extract name from line
            line = f.readline()

            if "Executives" in line or "Participants" in line:
                break


            count += 1
            if count > 100:
                return

        count = 0
        while True:

            # Extract name from line
            line = f.readline()
            name = line.split("-")[0].strip()

            if "Analysts" in line or "Participants" in line:
                break

            executives.append(name)

            count += 1
            if count > 100:
                return

        count = 0
        while True:

            # Extract name from line
            line = f.readline()
            name = line.split("-")[0].strip()

            if "Operator" in line:
                break

            analysts.append(name)

            count += 1
            if count > 100:
                return

        # Extract body of text now
        body = list()

        count = 0
        while True:

            # Extract name from line
            line = f.readline().strip()

            flag = False

            count += 1
            if count > 100:
                return

            for name in executives:
                if name in line:
                    flag = True
                    break

            if flag:
                continue

            if ("Question-and-Answer" in line or "Questions-and-Answer" in line) and len(line) < 100:
                break

            body.append(line + " ")




        questions = list()
        while True:

            line = f.readline().strip()

            if not line:
                break

            for name in analysts:
                if name in line:
                    question = f.readline().strip()

                    if '?' in question:
                        questions.append(question)
                    break


        body = "".join(body)

        # Return body and one random question
        return (body.lower(), random.choice(questions).lower()) if body and questions else None

def readLangs():
    print("Reading lines...")

    pairs = list()
    import os

    with os.scandir('cleaned_scrape/') as entries:
        for entry in entries:
            if entry.path.endswith(".txt"):

                pair = find_pairs(entry)
                if pair:
                    pairs.append(pair)
                    break

    input_lang = Lang("input")
    output_lang = Lang("output")

    return input_lang, output_lang, pairs

vectors = bcolz.open(f'{glove_path}/6B.50.dat')[:]
words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))

glove = {w: vectors[word2idx[w]] for w in words}


def find_matrix(input_lang):
    matrix_len = input_lang.n_words

    weights_matrix = torch.zeros((matrix_len, 50))
    words_found = 0

    for i, word in enumerate(input_lang.word2index):
        try:
            weights_matrix[i] = torch.from_numpy(glove[word])
            words_found += 1
        except KeyError:
            weights_matrix[i] = torch.from_numpy(np.random.normal(scale=0.6, size=(50, )))

    return weights_matrix

def prepareData():

    input_lang, output_lang, pairs = readLangs()
    print("Read %s sentence pairs" % len(pairs))

    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    print("Building GLoVE embedding:")
    input_weights_matrix = find_matrix(input_lang)
    output_weights_matrix = find_matrix(output_lang)
    print("GLoVE embedding built")
    return input_lang, output_lang, pairs, input_weights_matrix, output_weights_matrix

input, output, pairs, input_weights_matrix, output_weights_matrix = prepareData()

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input, pair[0])
    target_tensor = tensorFromSentence(output, pair[1])
    return (input_tensor, target_tensor)



if __name__ == "__main__":

    for input, output in pairs:
        print("Input: " + input)
        print("Output: " + output)
        print()
    # prepareData("cleaned_scrape/A_2017_Q1.txt")
