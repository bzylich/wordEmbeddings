import tensorflow as tf
import numpy as np
import collections
import os
import datetime as dt
import nltk

# code based on: https://github.com/adventuresinML/adventures-in-ml-code/blob/master/lstm_tutorial.py

def buildDataset(words, vocabularySize):
    # preprocess the text to break up sentences and remove book structure information
    newWords = []
    eosExceptions = ['Mr', 'Mrs', 'Ms', 'PhD']
    bookStructure = ['VOLUME', 'CHAPTER']
    eosPunctuation = ['.']#, '!', '?']
    # construct sentences
    sentences = []
    currentSentence = []
    removedPhrase = []
    inBrackets = False
    foundBookStructure = False
    for i in range(len(words)):
        if words[i] in eosPunctuation and (words[i-1] not in eosExceptions):
            currentSentence.append('<eos>')
            sentences.append(currentSentence)
            currentSentence = []
            newWords.append('<eos>')
        elif i == len(words) - 1:
            currentSentence.append(words[i])
            currentSentence.append('<eos>')
            sentences.append(currentSentence)
            newWords.append(words[i])
            newWords.append('<eos>')
        elif words[i] == ']' and inBrackets:
            inBrackets = False
            removedPhrase.append(']')
            # print('removed phrase: ')
            # print(removedPhrase)
            removedPhrase = []
        elif words[i] == '[' or inBrackets:
            inBrackets = True
            removedPhrase.append(words[i])
        elif words[i] in bookStructure:
            foundBookStructure = True
        elif foundBookStructure:
            foundBookStructure = False
            # print('Removed:', words[i-1], words[i])
        else:
            currentSentence.append(words[i])
            newWords.append(words[i])

    words = newWords

    # for s in sentences:
    #     print(s)
    #     input()
    # sentence = ''
    # for w in words:
    #     if w == '<eos>':
    #         print(sentence + '.')
    #         sentence = ''
    #         input()
    #     else:
    #         sentence += w + ' '



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
    return trainData, validationData, testData, index2Word

class Input(object):

    def __init__(self, batchSize, numSteps, data):
        self.batchSize = batchSize
        self.numSteps = numSteps
        self.epochSize = ((len(data) // batchSize) - 1) // numSteps
        self.inputData, self.targets = generateBatch(data, batchSize, numSteps)

class LanguageModel(object):

    def __init__(self, input, isTraining, hiddenSize, vocabularySize, numLayers, dropout=0.5, initScale=0.05):
        self.isTraining = isTraining
        self.input = input
        self.batchSize = input.batchSize
        self.numSteps = input.numSteps
        self.hiddenSize = hiddenSize

        # create word embeddings
        with tf.device("/cpu:0"):
            embedding = tf.Variable(tf.random_uniform([vocabularySize, self.hiddenSize], -initScale, initScale))
            inputs = tf.nn.embedding_lookup(embedding, self.input.inputData)

        # add dropout regularization
        if isTraining and dropout < 1:
            inputs = tf.nn.dropout(inputs, dropout)

        self.initState = tf.placeholder(tf.float32, [numLayers, 2, self.batchSize, self.hiddenSize])

        statePerLayerList = tf.unstack(self.initState, axis=0)
        rnnTupleState = tuple([tf.contrib.rnn.LSTMStateTuple(statePerLayerList[i][0], statePerLayerList[i][1]) for i in range(numLayers)])

        cell = tf.contrib.rnn.LSTMCell(hiddenSize, forget_bias=1.0)

        # add dropout wrapper if training
        if isTraining and dropout < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
        if numLayers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(numLayers)], state_is_tuple=True)

        output, self.state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, initial_state=rnnTupleState)
        output = tf.reshape(output, [-1, hiddenSize])

        softmax_W = tf.Variable(tf.random_uniform([hiddenSize, vocabularySize], -initScale, initScale))
        softmax_b = tf.Variable(tf.random_uniform([vocabularySize], -initScale, initScale))
        logits = tf.nn.xw_plus_b(output, softmax_W, softmax_b)

        # reshape logits to be 3D tensor for sequence loss
        logits = tf.reshape(logits, [self.batchSize, self.numSteps, vocabularySize])

        loss = tf.contrib.seq2seq.sequence_loss(logits, self.input.targets, tf.ones([self.batchSize, self.numSteps], dtype=tf.float32), average_across_timesteps=False, average_across_batch=True)

        # update cost
        self.cost = tf.reduce_sum(loss)

        # get prediction accuracy
        self.softmaxOut = tf.nn.softmax(tf.reshape(logits, [-1, vocabularySize]))
        self.predict = tf.cast(tf.argmax(self.softmaxOut, axis=1), tf.int32)
        correctPrediction = tf.equal(self.predict, tf.reshape(self.input.targets, [-1]))
        self.accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

        if not isTraining:
            return

        self.learningRate = tf.Variable(0.0, trainable=False)
        tVars = tf.trainable_variables()
        gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tVars), maxGradientNorm)
        optimizer = tf.train.GradientDescentOptimizer(self.learningRate)
        self.trainOp = optimizer.apply_gradients(zip(gradients, tVars), global_step=tf.train.get_or_create_global_step())

        self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self.lr_update = tf.assign(self.learningRate, self.new_lr)

    def assignLearningRate(self, session, lr):
        session.run(self.lr_update, feed_dict={self.new_lr: lr})

def train(trainData, vocabularySize, numLayers, numEpochs, batchSize, modelSaveName, lr=1.0, maxLREpoch=10, lrDecay=0.93, printIterations=50):
    trainInput = Input(batchSize=batchSize, numSteps=35, data=trainData)
    trainModel = Model(trainInput, isTraining=True, hiddenSize=650, vocabularySize=vocabularySize, numLayers=numLayers)
    initOp = tf.global_variables_initializer()
    initialDecay = lrDecay
    with tf.Session() as session:
        sess.run([initOp])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver = tf.train.Saver()
        for epoch in range(numEpochs):
            lrDecayNew = initialDecay ** max(epoch + 1 - maxLREpoch, 0.0)
            trainModel.assignLearningRate(session, learningRate * lrDecayNew)

            currentState = np.zeros((numLayers, 2, batchSize, trainModel.hiddenSize))
            currentTime = dt.datetime.now()
            for step in range(trainInput.epochSize):
                if step % printIterations != 0:
                    cost, _, currentState = session.run([trainModel.cost, trainModel.trainOp, trainModel.state], feed_dict={trainModel.initState: currentState})
                else:
                    seconds = (float((dt.datetime.now() - currentTime).seconds()) / printIterations)
                    currentTime = dt.datetime.now()
                    cost, _, currentState, accuracy = session.run([trainModel.cost, trainModel.trainOp, trainModel.state, trainmodel.accuracy], feed_dict={trainModel.initState: currentState})
                    print("Epoch", epoch, "Step", step, "cost:", cost, "accuracy:", accuracy, "seconds per step:", seconds)

            # save checkpoint
            saver.save(session, './rnn_log/' + modelSaveName, global_step=epoch)
        # final save
        saver.save(session, './rnn_log/' + modelSaveName + '-final')
        coord.request_stop()
        coord.join(threads)

def test(modelPath, testData, index2Word):
    testInput = Input(batchSize=20, numSteps=35, data=testData)
    testModel = Model(testInput, isTraining=False, hiddenSize=650, vocabularySize=len(index2Word), numLayers=2)
    saver = tf.train.Saver()

    with tf.Session() as session:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        currentState = np.zeros((2, 2, testModel.batchSize, testModel.hiddenSize))
        # restore trained model
        saver.restore(session, './rnn_log/' + modelPath)

        # get average accuracy over numAccBatches
        numAccBatches = 30
        checkBatchIndex = 25
        accCheckThreshold = 5
        accuracy = 0
        for batch in range(numAccBatches):
            if batch == checkBatchIndex:
                trueValues, predictions, currentState, accuracy = session.run([testModel.targets, testModel.predict, testModel.state, testModel.accuracy], feed_dict={testModel.initState: currentState})
                predictionString = [index2Word[x] for x in predictions[:testModel.numSteps]]
                trueValuesString = [index2Word[x] for x in trueValues[0]]
                print('True values (1st line) vs predicted values (2nd line):')
                print(' '.join(trueValuesString))
                print(' '.join(predictionString))
            else:
                acc, currentState = session.run([testModel.accuracy, testModel.state], feed_dict={testModel.initState: currentState})
            if batch >= accCheckThreshold:
                accuracy += acc
        print("Average accuracy:", (accuracy / (numAccBatches - accCheckThreshold)))
        coord.request_stop();
        coord.join(threads);

if __name__ == "__main__":
    vocabularySize = 5000

    print('loading dataset...')
    emma = nltk.corpus.gutenberg.words('austen-emma.txt')
    # print(list(emma))
    print('preparing dataset...')
    trainData, validationData, testData, index2Word = buildDataset(emma, vocabularySize)
    #
    # # train
    #
    # train(trainData, vocabularySize, numLayers=2, numEpochs=60, batchSize=20, modelSaveName='two-layer-lstm-60-epoch')
    #
    # # test
    #
    # test('two-layer-lstm-60-epoch', testData, index2Word)
