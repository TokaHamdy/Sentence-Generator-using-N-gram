import random
from nltk.corpus import brown
from nltk.tokenize import word_tokenize


# Preprocessing
def preprocess_corpus(corpus):
    processed_corpus = []
    for sentence in corpus:
        # Covert sentence from a list into string.
        new_sentence = " ".join(sentence)

        # Tokenize sentences into words (Word Tokenization).
        tokens = word_tokenize(new_sentence)

        # Remove punctuation marks from tokens.
        tokens = [token for token in tokens if token.isalpha()]

        # Convert all tokens to lowercase
        tokens = [token.lower() for token in tokens]

        processed_corpus.append(tokens)
    return processed_corpus


# Build N-gram model
def build_ngram_model(corpus, n):
    # To store the dictionary.
    ngram_model = {}
    for sentence in corpus:
        for i in range(len(sentence) - n + 1):
            # Creating a tuple of n-1 words to use it as a key.
            ngram = tuple(sentence[i:i + n - 1])
            next_word = sentence[i + n - 1]

            # If this ngram exists to avoid access out of bound.
            if ngram in ngram_model:
                ngram_model[ngram].append(next_word)
            else:
                ngram_model[ngram] = [next_word]
    return ngram_model


# Sentence Generator
def generate_sentence(ngram_model, n, max_len):
    # Generate random key.
    ngram = random.choice(list(ngram_model.keys()))
    sentence = list(ngram)

    while len(sentence) < max_len:
        if ngram in ngram_model:
            next_word = random.choice(ngram_model[ngram])
            sentence.append(next_word)
            # creating a tuple from last (n-1)words form the sentence.
            ngram = tuple(sentence[-(n - 1):])
        else:
            # unable to find the next word.
            break
    return " ".join(sentence)


# Main function
def work(m, n, max_len, corpus):
    # Import brown corpus
    brown_corpus = brown.sents(categories=corpus)
    # print(brown.categories())

    # Doing a preprocessing
    processed_corpus = preprocess_corpus(brown_corpus)

    # Build the N-gram model
    ngram_model = build_ngram_model(processed_corpus, n)

    # Generate m sentences
    sentences = []
    for _ in range(m):
        sentence = generate_sentence(ngram_model, n, max_len)
        sentences.append(sentence)

    return sentences


# Test the program
m = int(input("Enter the number of sentences: "))
n = int(input("Enter 2 for bigram, 3 for trigram: "))
max_len = int(input("Enter the maximum number of words in a sentence: "))

corpus = 'adventure'

c = 1
sentences = work(m, n, max_len, corpus)

for sentence in sentences:
    print(f"Sentence {c}: {sentence}")
    c += 1
