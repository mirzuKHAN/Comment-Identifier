import math
from util import *


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def extractWordFeatures(x):
    d = dict.fromkeys(x.split(), 0)
    for w in x.split():
        d[w] += 1
    return d


def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    weights = {}

    def predict(x):
        f = featureExtractor(x)
        if dotProduct(weights, f) < 0.0:
            return -1
        else:
            return 1

    for t in range(numIters):
        for example in trainExamples:
            x, y = example
            f = featureExtractor(x)
            margin = dotProduct(weights, f) * y
            if margin < 1:
                increment(weights, eta * y, f)
        print("Iteration:%s, Training error:%s, Test error:%s " % (t, evaluatePredictor(trainExamples, predict), evaluatePredictor(testExamples, predict)))
    return weights


trainExamples = readExamples('polarity.train')
devExamples = readExamples('polarity.dev')
featureExtractor = extractWordFeatures
weights = learnPredictor(trainExamples, devExamples, featureExtractor, numIters=20, eta=0.01)
outputWeights(weights, 'weights')
outputErrorAnalysis(devExamples, featureExtractor, weights, 'error-analysis')
trainError = evaluatePredictor(trainExamples, lambda x : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
devError = evaluatePredictor(devExamples, lambda x : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
print(("Official: train error = %s, dev error = %s" % (trainError, devError)))