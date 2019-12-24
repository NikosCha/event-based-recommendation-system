import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from figures.diagrams import create_diagram
from utils.variables import get_variable, init_variable


class TemporalModel:
    def __init__(self, graph):
        self.graph = graph
        self.session = tf.compat.v1.Session(config=None, graph=graph)

    def get_cosine_similarity(self, x1, x2, name='Cosine_loss'):
        with tf.name_scope(name):
            x1_val = tf.sqrt(tf.reduce_sum(tf.matmul(x1,tf.transpose(x1)),axis=1))
            x2_val = tf.sqrt(tf.reduce_sum(tf.matmul(x2,tf.transpose(x2)),axis=1))
            denom =  tf.multiply(x1_val,x2_val)
            num = tf.reduce_sum(tf.multiply(x1,x2),axis=1)
            return tf.compat.v1.div(num,denom)


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

            randomUserTesting = testingData[testingData.user_id == userID]
            #get one random event of user's testing set
            randomUserTesting = randomUserTesting.sample(n=1,replace=False)
            #random event 
            randomEvent = trainingData.sample(n=1,replace=False)


            scoreKnown = self.get_score()
            scoreUnknown = self.get_score()

            diff.append((scoreKnown - scoreUnknown) > 0)

            
        diff = np.asarray(diff)
        auc = np.mean(diff == True)
        print('----SOCIAL MODEL VALIDATION----')
        print(auc)
        return auc

    def get_score(self):
        score = ''
        return score

    def make_recommendation(self, event_dictionary, user_id=None, num_items=10):

        return ''    

    def __del__(self):
        tf.compat.v1.reset_default_graph()
        print ("deleted")
