#imports
import numpy as np
import pandas as pd
import random
import re
from gensim.parsing.preprocessing import remove_stopwords
import nltk
nltk.download('words')
words = set(nltk.corpus.words.words())


###Convert input data into a list of lists, with each line of data being a nested list
###Also generates a list of labels based on if any of the six toxic categories are fulfilled
def convertData(data):
    i = 0
    l = len(list(data.itertuples()))
    labels = [-1]*l
    print("Cleaning data")
    printProgressBar(0, l, prefix= 'Progress:', suffix='Complete', length=50) # print initial state of progress bar
    for row in data.itertuples():
        temp = row.comment_text
        temp = temp.replace("\n", " ")
        temp = clean_tweet(temp)
        if temp != '':
            data.iat[i, 1] = temp
        if row.toxic == 1 or row.severe_toxic == 1 or row.obscene == 1 or row.threat == 1 or row.insult == 1 or row.identity_hate == 1:
            labels[i] = 1
        printProgressBar(i+1, l, prefix= 'Progress:', suffix='Complete', length=50) # update progress bar 
        i+=1

    return data, labels


# function which will drive the machine learning algo
def standardPerceptron(traindata, badWords, f):

    ######Preprocessing######
    traindata, trainlabels = convertData(traindata)
    ######Form the vocabulary######
    vocabulary = []
    l = len(list(traindata.itertuples()))
    i = 0
    print("Building vocabulary")
    """ temp = badWords.readline()
    while(temp): #append each bad word to a list
        temp = temp.rstrip('\n')
        vocabulary.append(temp)
        temp = badWords.readline() """
    printProgressBar(0, l, prefix= 'Progress:', suffix='Complete', length=50) # print initial state of progress bar
    for line in traindata.itertuples():
        temp = line.comment_text
        temp = temp.split(' ')
        for word in temp:
            vocabulary.append(word)
        printProgressBar(i+1, l, prefix= 'Progress:', suffix='Complete', length=50) # update progress bar 
        i+=1
    vocabulary = list(set(vocabulary))
    vocabulary.sort()


    ######Convert training data into a list of features######
    M = len(vocabulary)
    featuresSet = []
    i = 0
    print("Building feature vectors")
    printProgressBar(0, l, prefix= 'Progress:', suffix='Complete', length=50) # print initial state of progress bar
    for line in traindata.itertuples():
        temp = line.comment_text
        feature = [0]*M
        temp = temp.split(' ')
        for word in temp:
            if word in vocabulary:
                feature[vocabulary.index(word)] = 1
        featuresSet.append(feature)
        printProgressBar(i+1, l, prefix= 'Progress:', suffix='Complete', length=50) # update progress bar 
        i += 1


    ######Online Binary-Classifier Learning Algorithms######

    ######Standard Perceptron######
    w = [0] * M
    learningRate = 1
    numMistakes = [0]*20
    
    wlen = len(w)
    flen = len(featuresSet)
    l = 20
    print("Creating prediction model")
    printProgressBar(0, l, prefix= 'Progress:', suffix='Complete', length=50) # print initial state of progress bar
    for itr in range(20):
        
        #shuffle training data
        temp = list(zip(featuresSet, trainlabels))
        random.shuffle(temp)
        res1, res2 = zip(*temp)
        featuresSet, trainlabels = list(res1), list(res2)

        for i in range(flen): #training example -> (featureSet line = xt, trainlabels number = yt)
            
            #predict using the current weights
            temp1 = np.array(featuresSet[i])
            temp2 = np.array(w)
            dotProd = np.dot(temp1, temp2) 
            prediction = np.sign(dotProd)

            #prediction <= 0 -> mistake. update the weights
            if prediction <= 0: 
                numMistakes[itr] += 1 #increment number of mistakes for the current iteration
                for j in range(wlen):
                    w[j] = w[j] + learningRate * (trainlabels[i] - dotProd) * featuresSet[i][j]
                learningRate += 0.1
        printProgressBar(itr+1, l, prefix= 'Progress:', suffix='Complete', length=50) # update progress bar

        numerator = numMistakes[itr]
        print("iteration-", itr+1, " no-of-mistakes=", numMistakes[itr], " training-accuracy=", 1 - (numerator / len(featuresSet)), file=f)

    return w, vocabulary, f



def averagedPerceptron(stoplist, traindata, trainlabels, f):
    ######Preprocessing######
    stoplist = convertData(stoplist)
    traindata = convertData(traindata)
    trainlabels = convertData(trainlabels)
    trainlabels = list(map(int, trainlabels))

    for i in range(len(trainlabels)):
        if trainlabels[i] == 0:
            trainlabels[i] = -1

    ######Form the vocabulary######
    vocabulary = []
    for line in traindata:
        temp = line.split(' ')
        for word in temp:
            if word not in stoplist and word not in vocabulary:
                vocabulary.append(word)
    vocabulary.remove('')
    vocabulary.sort()


    ######Convert training data into a list of features######
    M = len(vocabulary)
    featuresSet = []
    i = 0
    for line in traindata:
        featuresSet.append([0]*M)
        temp = line.split(' ')
        for word in temp:
            if word in vocabulary:
                featuresSet[i][vocabulary.index(word)] = 1
        i += 1


    ######Online Binary-Classifier Learning Algorithms######

    ######Averaged Perceptron######
    w = [0] * M
    learningRate = 1
    cm = 1
    numMistakes = [0]*20
    wList = []

    l = 20
    printProgressBar(0, l, prefix= 'Progress:', suffix='Complete', length=50) # print initial state of progress bar
    for itr in range(20):
        #shuffle training data
        temp = list(zip(featuresSet, trainlabels))
        random.shuffle(temp)
        res1, res2 = zip(*temp)
        featuresSet, trainlabels = list(res1), list(res2)

        for i in range(len(featuresSet)): #training example -> (featureSet line = xt, trainlabels number = yt)
            
            #predict using the current weights
            temp1 = np.array(featuresSet[i])
            temp2 = np.array(w)
            dotProd = np.dot(temp1, temp2) 
            prediction = np.sign(dotProd)

            #prediction <= 0 -> mistake. update the weights
            if prediction <= 0: 
                numMistakes[itr] += 1 #increment number of mistakes for the current iteration
                for j in range(len(w)):
                    w[j] = w[j] + learningRate * (trainlabels[i] - dotProd) * featuresSet[i][j]
                wList.append((w, cm))
                cm = 1
                learningRate += 1
            else:
                cm += 1#keep track of which w survived the longest by associating a w with a cm
        
        printProgressBar(itr+1, l, prefix= 'Progress:', suffix='Complete', length=50) # update progress bar 

        numerator = numMistakes[itr]
        print("iteration-", itr+1, " no-of-mistakes=", numMistakes[itr], " training-accuracy=", 1 - (numerator / len(featuresSet)), file=f)        
        
    #print("Mistakes during training: ", numMistakes)

    #calculate average weight vector to pass to testing predictor
    data = []
    for (tup1, tup2) in wList:
        data.append([value * tup2 for value in tup1])
    w = list(np.average(data, axis=0))
    

    return w, vocabulary, f



def predictor(w, testdata, testlabels, stoplist, vocabulary, f):
    stoplist = convertData(stoplist)
    testdata = convertData(testdata)
    testlabels = convertData(testlabels)
    testlabels = list(map(int, testlabels))

    for i in range(len(testlabels)):
        if testlabels[i] == 0:
            testlabels[i] = -1

    ######Convert training data into a list of features######
    M = len(vocabulary)
    featuresSet = []
    i = 0
    for line in testdata:
        featuresSet.append([0]*M)
        temp = line.split(' ')
        for word in temp:
            if word in vocabulary:
                featuresSet[i][vocabulary.index(word)] = 1
        i += 1

    numMistakes = [0]*len(featuresSet)
    for itr in range(20):
        for i in range(len(featuresSet)):
            #predict using the current weights
            temp1 = np.array(featuresSet[i])
            temp2 = np.array(w)
            dotProd = np.dot(temp1, temp2) 
            prediction = np.sign(dotProd)
            if prediction != testlabels[i]:
                numMistakes[i] = 1

        #print("Number of mistakes during testing: ", numerator)
        #print("Testing accuracy: ", 1 - (numerator / len(numMistakes)))
        #print("iteration-", itr+1, " training-accuracy=", trainingAccuracyPerRun[i], " testing-accuracy=", testingAccuracyPerRun[i], file=f)
        numerator = numMistakes[itr]
        print("iteration-", itr+1, " no-of-mistakes=", numMistakes[itr], " testing-accuracy=", 1 - (numerator / len(featuresSet)), file=f)   

    return f

def ocrStandardPerceptron(ocr_train, f):
    vowels = ['a', 'e', 'i', 'o', 'u']
    consonants = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z']


    ######Clean data######
    dataList = []
    temp = ocr_train.readline()
    while(temp):
        temp = temp.rstrip('\n')
        temp = temp.split("\t")
        dataList.append(temp)
        temp = ocr_train.readline()

    featuresSet = []
    trainlabels = [] #use 1s to represent vowels, and -1s to represent consonants
    for line in dataList:
        if line[0] != '':
            temp = list(line[1][2:])
            temp = [int(char) for char in temp]
            featuresSet.append(temp)
            if line[2] in vowels:
                trainlabels.append(1)
            else:
                trainlabels.append(-1)


    ######Binary Classifier######
    ######Standard Perceptron######
    M = 128
    w = [0] * M
    learningRate = 1
    numMistakes = [0]*20

    l = 20
    printProgressBar(0, l, prefix= 'Progress:', suffix='Complete', length=50) # print initial state of progress bar
    for itr in range(20):
        
        #shuffle training data
        temp = list(zip(featuresSet, trainlabels))
        random.shuffle(temp)
        res1, res2 = zip(*temp)
        featuresSet, trainlabels = list(res1), list(res2)

        for i in range(len(featuresSet)): #training example -> (featureSet line = xt, trainlabels number = yt)
            
            #predict using the current weights
            temp1 = np.array(featuresSet[i])
            temp2 = np.array(w)
            dotProd = np.dot(temp1, temp2) 
            prediction = np.sign(dotProd)

            #prediction <= 0 -> mistake. update the weights
            if prediction <= 0: 
                numMistakes[itr] += 1 #increment number of mistakes for the current iteration
                for j in range(len(w)):
                    w[j] = w[j] + learningRate * (trainlabels[i] - dotProd) * featuresSet[i][j]
                learningRate += 1
        
        printProgressBar(itr+1, l, prefix= 'Progress:', suffix='Complete', length=50) # update progress bar         
        numerator = numMistakes[itr]
        print("iteration-", itr+1, " no-of-mistakes=", numMistakes[itr], " training-accuracy=", 1 - (numerator / len(featuresSet)), file=f)
    #print("Mistakes during training: ", numMistakes)
    
    return w, f

def ocrAveragedPerceptron(ocr_train, f):
    ######Preprocessing######
    vowels = ['a', 'e', 'i', 'o', 'u']
    consonants = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z']

    ######Clean data######
    dataList = []
    temp = ocr_train.readline()
    while(temp):
        temp = temp.rstrip('\n')
        temp = temp.split("\t")
        dataList.append(temp)
        temp = ocr_train.readline()

    featuresSet = []
    trainlabels = [] #use 1s to represent vowels, and -1s to represent consonants
    for line in dataList:
        if line[0] != '':
            temp = list(line[1][2:])
            temp = [int(char) for char in temp]
            featuresSet.append(temp)
            if line[2] in vowels:
                trainlabels.append(1)
            else:
                trainlabels.append(-1)


    ######Online Binary-Classifier Learning Algorithms######

    ######Binary Classifier######
    ######Averaged Perceptron######
    M = 128
    w = [0] * M
    cm = 1
    learningRate = 1
    numMistakes = [0]*20
    wList = []

    l = 20
    printProgressBar(0, l, prefix= 'Progress:', suffix='Complete', length=50) # print initial state of progress bar
    for itr in range(20):
        
        #shuffle training data
        temp = list(zip(featuresSet, trainlabels))
        random.shuffle(temp)
        res1, res2 = zip(*temp)
        featuresSet, trainlabels = list(res1), list(res2)

        for i in range(len(featuresSet)): #training example -> (featureSet line = xt, trainlabels number = yt)
            
            #predict using the current weights
            temp1 = np.array(featuresSet[i])
            temp2 = np.array(w)
            dotProd = np.dot(temp1, temp2) 
            prediction = np.sign(dotProd)

            #prediction <= 0 -> mistake. update the weights
            if prediction <= 0: 
                numMistakes[itr] += 1 #increment number of mistakes for the current iteration
                for j in range(len(w)):
                    w[j] = w[j] + learningRate * (trainlabels[i] - dotProd) * featuresSet[i][j]
                wList.append((w, cm))
                cm = 1
                learningRate += 1
            else:
                cm += 1#keep track of which w survived the longest by associating a w with a cm
        
        printProgressBar(itr+1, l, prefix= 'Progress:', suffix='Complete', length=50) # update progress bar   
        numerator = numMistakes[itr]
        print("iteration-", itr+1, " no-of-mistakes=", numMistakes[itr], " training-accuracy=", 1 - (numerator / len(featuresSet)), file=f)      
        
    #print("Mistakes during training: ", numMistakes)

    #calculate average weight vector to pass to testing predictor
    data = []
    for (tup1, tup2) in wList:
        data.append([value * tup2 for value in tup1])
    w = list(np.average(data, axis=0))

    return w, f

def ocrPredictor(ocr_test, w, f):
    vowels = ['a,', 'e', 'i', 'o', 'u']
    consonants = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z']

    ######Clean data######
    dataList = []
    temp = ocr_test.readline()
    while(temp):
        temp = temp.rstrip('\n')
        temp = temp.split("\t")
        dataList.append(temp)
        temp = ocr_test.readline()

    featuresSet = []
    testlabels = [] #use 1s to represent vowels, and -1s to represent consonants
    for line in dataList:
        if line[0] != '':
            temp = list(line[1][2:])
            temp = [int(char) for char in temp]
            featuresSet.append(temp)
            if line[2] in vowels:
                testlabels.append(1)
            else:
                testlabels.append(-1)

    numMistakes = [0]*len(featuresSet)
    l = 20
    printProgressBar(0, l, prefix= 'Progress:', suffix='Complete', length=50) # print initial state of progress bar
    for itr in range(20):
        for i in range(len(featuresSet)):
            #predict using the current weights
            temp1 = np.array(featuresSet[i])
            temp2 = np.array(w)
            dotProd = np.dot(temp1, temp2) 
            prediction = np.sign(dotProd)
            if prediction < 0 and testlabels[i] == 1: #should be a vowel, but predicted consonant
                numMistakes[i] = 1
            elif prediction > 0 and testlabels[i] == 0: #should be a consonant, but predicted vowel
                numMistakes[i] = 1
                

        numerator = numMistakes[i]
        print("iteration-", itr+1, " no-of-mistakes=", numMistakes[i], " testing-accuracy=", 1 - (numerator / len(featuresSet)), file=f)    #print("Number of mistakes during testing: ", numerator)
        printProgressBar(itr+1, l, prefix= 'Progress:', suffix='Complete', length=50) # update progress bar         

    #print("Testing accuracy: ", 1 - (numerator / len(numMistakes)))
    return f

#website that provides a handy function for cleaning tweets
#https://catriscode.com/2021/05/01/tweets-cleaning-with-python/
def clean_tweet(tweet):
    if type(tweet) == float:
        return ""
    temp = tweet.lower()
    temp = re.sub("'", "", temp) # to avoid removing contractions in english
    temp = re.sub("@[A-Za-z0-9_]+","", temp)
    temp = re.sub("#[A-Za-z0-9_]+","", temp)
    temp = re.sub(r'http\S+', '', temp)
    temp = re.sub('[()!?]', ' ', temp)
    temp = re.sub('\[.*?\]',' ', temp)
    temp = re.sub("[^a-z0-9]"," ", temp)
    temp = re.sub(r'\d+', '', temp)
    temp = remove_stopwords(temp)
    temp = temp.split()
    tempList = []
    for w in temp:
        if w in words:
            tempList.append(w)
    temp = tempList
    #temp = [w for w in temp if not w in words.words()]
    temp = " ".join(word for word in temp)
    return temp

# Taken from https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters to make terminal output look nicer
# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def main():

############# import data from txt files #############################
    traindata = pd.read_csv("train.csv")#, nrows=int(159571/2))
    if not traindata.empty:
        print("loading file OK")
    else:
        print("loading file error")
        return
    badWords = open("bad-words.csv")
    if badWords:
        print("loading file OK")
    else:
        print("loading file error")
        return
###########################################################################
    
    
    outfile = open('output.txt', 'a')#open output.txt to print output data

    print("Toxic Comments Perceptron", file=outfile)
    #run standard perceptron
    print("\nTraining", file=outfile)
    w, vocabulary, outfile = standardPerceptron(traindata, badWords, outfile)
    #print("\nTesting", file=f)
    #outfile = predictor(w, testdata, testlabels, stoplist, vocabulary, outfile)

    
    return



main()