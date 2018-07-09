# This program learns word embeddings using tensorflow

import collections
import nltk
import math
import os
import sys
import random
from tempfile import gettempdir
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

vocabularySize = 5000

embeddingSize = 128 # dimension of the embedding vector
nceSampleSize = 64 # number of negative examples to sample
learningRate = 1.0
batchSize = 128
skipWindow = 1 # how many words to consider on the left and right
numSkips = 2 # how many times to reuse an input to generate a label
numSteps = 100000

# pick validation set from most frequent words
validationSize = 16 # size of random set of words to evaluate similarity on
validationWindow = 100 # only pick from this many most frequent words in the vocabulary
validationExamples = np.random.choice(validationWindow, validationSize, replace=False)

# process raw input and build vocabulary
def buildDataset(words, vocabularySize):
    word2Index = {}
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
    count[0] = ('<Unknown>', unknownCount)
    index2Word = dict(zip(word2Index.values(), word2Index.keys()))
    return data, count, word2Index, index2Word

dataIndex = 0

def generateBatch(batchSize, numSkips, skipWindow):
    global dataIndex
    assert batchSize % numSkips == 0
    assert numSkips <= 2 * skipWindow

    batch = np.ndarray(shape=(batchSize), dtype=np.int32)
    labels = np.ndarray(shape=(batchSize, 1), dtype=np.int32)
    span = 2 * skipWindow + 1 # skipWindow target skipWindow
    skipBuffer = collections.deque(maxlen=span)
    if dataIndex + span > len(data):
        dataIndex = 0
    skipBuffer.extend(data[dataIndex : dataIndex + span])
    dataIndex += span
    for i in range(batchSize // numSkips):
        contextWords = [w for w in range(span) if w != skipWindow]
        wordsToUse = random.sample(contextWords, numSkips)
        for j, contextWord in enumerate(wordsToUse):
            batch[i * numSkips + j] = skipBuffer[skipWindow]
            labels[i * numSkips + j, 0] = skipBuffer[contextWord]
        if dataIndex == len(data):
            skipBuffer.extend(data[0:span])
            dataIndex = span
        else:
            skipBuffer.append(data[dataIndex])
            dataIndex += 1
        # backtrack to avoid skipping words in the end of a batch
        dataIndex = (dataIndex + len(data) - span) % len(data)
    return batch, labels

# prepare our dataset
print('loading dataset...')
emma = nltk.corpus.gutenberg.words('austen-emma.txt')
print('preparing dataset...')
data, count, word2Index, index2Word = buildDataset(emma, vocabularySize)

# build and train skip-gram model
print('building model...')
graph = tf.Graph()

with graph.as_default():

    # input data
    with tf.name_scope('inputs'):
        # placeholders for inputs
        trainInputs = tf.placeholder(tf.int32, shape=[batchSize])
        trainLabels = tf.placeholder(tf.int32, shape=[batchSize, 1])
        validationDataset = tf.constant(validationExamples, dtype=tf.int32)

    # pin operations to CPU that are missing GPU implementation
    with tf.device('/cpu:0'):
        # embeddings for inputs
        with tf.name_scope('embeddings'):
            # initialize embedding matrix to be uniform in the unit cube
            embeddings = tf.Variable(tf.random_uniform([vocabularySize, embeddingSize], -1.0, 1.0))
            # look up vector for each of the source words in the batch
            embeddedInputs = tf.nn.embedding_lookup(embeddings, trainInputs)

        # define weights and biases for each word in the vocabulary for use with noise-contrastive estimation loss (in terms of logistic regression model)
        with tf.name_scope('weights'):
            nceWeights = tf.Variable(tf.truncated_normal([vocabularySize, embeddingSize], stddev=(1.0 / math.sqrt(embeddingSize))))
        with tf.name_scope('biases'):
            nceBiases = tf.Variable(tf.zeros([vocabularySize]))

    # compute NCE loss using a sample of the negative labels in each batch
    # NCE loss: http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nceWeights, biases=nceBiases, labels=trainLabels, inputs=embeddedInputs, num_sampled=nceSampleSize, num_classes=vocabularySize))

    tf.summary.scalar('loss', loss)

    # use SGD to optimize and compute gradients
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(loss)

    # compute cosine similarity between minibatch examples and all embeddings
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalizedEmbeddings = embeddings / norm
    validationEmbeddings = tf.nn.embedding_lookup(normalizedEmbeddings, validationDataset)

    similarity = tf.matmul(validationEmbeddings, normalizedEmbeddings, transpose_b=True)

    mergedSummary = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

# begin training
print('starting to train...')
with tf.Session(graph=graph) as session:
    # writer = tf.summary.FileWriter('./log/', session.graph)
    try:
        saver.restore(session, './log/model.ckpt')
        print('pretrained embeddings restored!')
    except:
        # initialize all variables before using them
        init.run()

        averageLoss = 0

        for step in range(numSteps):
            batchInputs, batchLabels = generateBatch(batchSize, numSkips, skipWindow)

            feedDict = {trainInputs: batchInputs, trainLabels: batchLabels}

            runMetadata = tf.RunMetadata()

            _, summary, currentLoss = session.run([optimizer, mergedSummary, loss], feed_dict=feedDict, run_metadata=runMetadata)
            averageLoss += currentLoss

            # writer.add_summary(summary, step)
            # if step == (numSteps - 1):
            #     writer.add_run_metadata(runMetadata, 'step %d' % step)

            if step % 2000 == 0:
                if step > 0:
                    averageLoss /= 2000
                print('Average loss at step ', step, ': ', averageLoss)
                averageLoss = 0

            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(validationSize):
                    validationWord = index2Word[validationExamples[i]]
                    topK = 8 # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:topK + 1]
                    logStr = 'Nearest to "%s":' % validationWord
                    for k in range(topK):
                        closeWord = index2Word[nearest[k]]
                        logStr += ' ' + closeWord
                    print(logStr)

    finalEmbeddings = normalizedEmbeddings.eval()

    # write corresponding labels to the embeddings
    with open('./log/metdata.tsv', 'w') as f:
        for i in range(vocabularySize):
            f.write(index2Word[i] + '\n')

    saver.save(session, os.path.join('./log/model.ckpt'))
    print('training complete!')

# writer.close()

# visualize the embeddings

def plotWithLabels(denseEmbeddings, labels, filename):
    assert denseEmbeddings.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18)) # in inches
    for i, label in enumerate(labels):
        x, y = denseEmbeddings[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.savefig(filename)

print('visualizing embeddings...')
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
numToPlot = 500
denseEmbeddings = tsne.fit_transform(finalEmbeddings[:numToPlot, :])
labels = [index2Word[i] for i in range(numToPlot)]
plotWithLabels(denseEmbeddings, labels, './tsne.png')
