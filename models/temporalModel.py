import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from figures.diagrams import create_diagram
from utils.variables import get_variable, init_variable
from scipy import spatial


class TemporalModel:
    def __init__(self, graph):
        self.graph = graph
        self.session = tf.compat.v1.Session(config=None, graph=graph)

    def get_cosine_similarity(self, x1, x2, name='Cosine_loss'):
        cosSim = 1 - spatial.distance.cosine(x1, x2)
        return cosSim


    def validate_model(self, trainingData, testingData, samples):
        diff = []

        #get unique user's ids 
        #we get testing data because if a user exists in testing data, exists in training aswell 
        uniqueUsers = np.unique(testingData.user_id)

        for _ in range(0,samples):
            #get random user
            index = np.random.randint(len(uniqueUsers), size=1)

            #get the random user
            userID=uniqueUsers[index][0]
            randomUserTraining = trainingData[trainingData.user_id == userID] 
            randomUserTesting = testingData[testingData.user_id == userID]
            #get one random event of user's testing set
            randomUserTesting = randomUserTesting.sample(n=1,replace=False)
            #random event 
            randomEvent = trainingData.sample(n=1,replace=False)

            userVector = self.get_user_vector(randomUserTraining)
            eventVectorKnown = self.get_event_vector(randomUserTesting)
            eventVectorUnknown = self.get_event_vector(randomEvent)

            similarityKnown = self.get_cosine_similarity(userVector, eventVectorKnown)
            similarityUnknown = self.get_cosine_similarity(userVector, eventVectorUnknown)

            if(similarityKnown - similarityUnknown) == 0:
                continue    
            diff.append((similarityKnown - similarityUnknown) > 0)

            
        diff = np.asarray(diff)
        auc = np.mean(diff == True)
        print('----TEMPORAL MODEL VALIDATION----')
        print(auc)
        return auc

    def get_score(self):
        score = ''
        return score

    def get_user_vector(self, data):
        userVector = np.ones(24*7)/10

        for _, row in data.iterrows():
            weekday = int(row.weekday)
            time = int(row.time)
            userVector[(weekday-1)*24 + time] = userVector[(weekday-1)*24 + time] + 1 

        userVector = userVector/len(data)
        
        return userVector

    def get_event_vector(self, data):
        eventVector = np.zeros(24*7)

        for _, row in data.iterrows():
            weekday = int(row.weekday)
            time = int(row.time)
            eventVector[(weekday-1)*24 + time] = eventVector[(weekday-1)*24 + time] + 1 

        return eventVector

    def make_recommendation(self, event_dictionary, user_id=None, num_items=10):

        return ''    

    def __del__(self):
        tf.compat.v1.reset_default_graph()
        print ("deleted")
