# This program learns word embeddings using tensorflow

import nltk
import math
import numpy as np
import tensorflow as tf

vocabularySize = 1000

embeddingSize = 128
nceSampleSize = 64
learningRate = 1.0
batchSize = 128
skipWindow = 1


# process raw input and build vocabulary
def buildDataset(words, vocabularySize):
    word2Index = {'<Unknown>': 0}
    count = [('<Unknown>', 0)]
    freqDist = nltk.FreqDist(words)
    count.extend(freqDist.most_common(vocabularySize - 1))

    for word, _ in count:
        word2Index[word] = len(word2Index)

    data = []
    unknownCount = 0
    for word in words:
        index = word2Index.get(word, 0)
        if index == 0:
            unknownCount += 1
        data.append(index)
    count[0][1] = unknownCount
    index2Word = dict(zip(word2Index.values(), word2Index.keys()))
    return data, count, word2Index, index2Word

def generateBatch(batchSize, numSkips, skipWindow):
    assert batchSize % numSkips == 0
    assert numSkips <= 2 * skipWindow
    

# prepare our dataset
emma = nltk.corpus.gutenberg.words('austen-emma.txt')
data, count, word2Index, index2Word = buildDataset(emma, vocabularySize)

# initialize embedding matrix to be uniform in the unit cube
embeddings = tf.Variable(tf.random_uniform([vocabularySize, embeddingSize], -1.0, 1.0))

# define weights and biases for each word in the vocabulary for use with noise-contrastive estimation loss (in terms of logistic regression model)
nceWeights = tf.Variable(tf.truncated_normal([vocabularySize, embeddingSize], stddev=(1.0 / math.sqrt(embeddingSize))))
nceBiases = tf.Variable(tf.zeros([vocabularySize]))

# placeholders for inputs
trainInputs = tf.placeholder(tf.int32, shape=[batchSize])
trainLabels = tf.placeholder(tf.int32, shape=[batchSize, 1])

# look up vector for each of the source words in the batch
embeddedInputs = tf.nn.embedding_lookup(embeddings, trainInputs)

# compute NCE loss using a sample of the negative labels in each batch
loss = tf.reduce_mean(tf.nn.nce_loss(weights=nceWeights, biases=nceBiases, labels=trainLabels, inputs=embeddedInputs, num_sampled=nceSampleSize, num_classes=vocabularySize))

# use SGD to optimize and compute gradients
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(loss)

for inputs, labels in generateBatch():
    feedDict = {trainInputs: inputs, trainLabels: labels}
    _, currentLoss = session.run([optimizer, loss], feed_dict=feedDict)
