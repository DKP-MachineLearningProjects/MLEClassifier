import numpy as np
import pandas as pd
from numpy.random import RandomState
import math

#'pima.csv' file contains column labels as x1 to x8 for features and 
#y for output
#Only x2, x3, x4, and y features are selected and split into train and test as dataFrames   
def SplitData():
    df = pd.read_csv("pima.csv")
    df=df.drop(columns=['x1','x5','x6','x7','x8'])
    rng = RandomState()

    train = df.sample(frac=0.5, random_state=rng)
    test = df.loc[~df.index.isin(train.index)]
    return (train, test)

#it predicts the posterior for each row of test data
#and calculate whether it classifies correctly or not
#by counting in correct and wrong varaible
def MLEClassifier(train ,test):
    correct=0
    wrong=0
    
    #split the given training samles with class 0 and class 1 groups
    #and convert them into lists
    train0=train[train['y']==0].values.tolist()
    train1=train[train['y']==1].values.tolist()
    test=test.values.tolist()

    #find the mean and varaince from train0 and train1 samples row wise
    mean0=np.mean(train0, axis=0)
    var0=np.var(train0, axis=0)
    mean1=np.mean(train1, axis=0)
    var1=np.var(train1, axis=0)

    #calculate the prior of class 0 training samples and class 1 traing samples
    prior0=len(train0)
    prior1=len(train1)

    #for each samples in test, calculate the posterior probability and 
    #find the predicted class and calculate wheather it is correctly classified or not
    for row in test:
        likelihood0=1.0
        likelihood1=1.0
        for i in range(3):
            likelihood0  *=  ( (1.0/(math.sqrt(2*3.1415*var0[i]))) * math.exp((-0.5)*((row[i]-mean0[i])**2)/var0[i]))
            likelihood1  *=  ( (1.0/(math.sqrt(2*3.1415*var1[i]))) * math.exp((-0.5)*((row[i]-mean1[i])**2)/var1[i]))
        posterior0=prior0 * likelihood0
        posterior1=prior1 * likelihood1
        #check whether given sample is correctly classified or not
        if(posterior0 > posterior1 and row[3]==0):
            correct+=1
        elif(posterior0 < posterior1 and row[3]==1):
            correct+=1
        else:
            wrong+=1
    return(correct,wrong)


listAccuracy=[]

#Iterate the classifier for 10 time as required by question
for i in range(10):
    train,test=SplitData()
    correct, wrong=MLEClassifier(train,test)
    listAccuracy.append(float(correct)/(correct+wrong)*100)

#calculate and print the statistics of MLEClassifier for the given data stored in 'pima.csv' (please look at the sample at bottom of the program)
meanAccuracy=np.average(listAccuracy)
sd=np.std(listAccuracy)
print "List of Accuracy for correct classification in percentage for 10 iteration\n", listAccuracy
print "Mean accuracy=", meanAccuracy
print "Standard Deviation=",sd 

#Ramdom OUTPUT
#List of Accuracy for correct classification in percentage for 10 iteration
# [74.47916666666666, 77.60416666666666, 74.47916666666666, 77.60416666666666, 74.21875, 75.52083333333334, 75.78125, 75.52083333333334, 76.5625, 71.875]
# Mean accuracy= 75.36458333333333
# Standard Deviation= 1.6354498404278717

#'pima.csv' file outline
# x1,x2,x3,x4,x5,x6,x7,x8,y
# 6,148,72,35,0,33.6,0.627,50,1
# 1,85,66,29,0,26.6,0.351,31,0
# 8,183,64,0,0,23.3,0.672,32,1
# 1,89,66,23,94,28.1,0.167,21,0
# 0,137,40,35,168,43.1,2.288,33,1