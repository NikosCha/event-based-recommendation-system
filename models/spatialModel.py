import datetime
import numpy as np
import pandas as pd
import tensorflow.compat.v2 as tf
from tqdm import tqdm
from bs4 import BeautifulSoup
from figures.diagrams import create_diagram
from utils.variables import get_variable, init_variable

tf.enable_v2_behavior()

class SpatialModel:
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

    def get_distance_between_two_points(self, lat1, lat2, lon1, lon2):
        from math import sin, cos, sqrt, atan2, radians

        #radius of earth    
        R = 6371.0
        dlon = float(lon2) - float(lon1)
        dlat = float(lat2) - float(lat1)
        
        a = sin(dlat / 2)**2 + cos(float(lat1)) * cos(float(lat2)) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = R * c
        return distance/1000 

    # latitude and longtitude Y are the user's lat and long and X are the event's coordinates

    def validate_model(self, trainingData, testingData, samples):
        diff = []

        #get unique user's ids 
        #we get testing data because if a user exists in testing data, exists in training aswell 
        uniqueUsers = np.unique(testingData.user_id)

        for _ in range(0,samples):
            #get random user
            index =  np.random.randint(len(uniqueUsers), size=1)

            #get the random user
            userID=uniqueUsers[index][0]
            randomUserTraining = trainingData[trainingData.user_id == userID]  
            randomUserTesting = testingData[testingData.user_id == userID]

            #get one random event of user's testing set
            randomUserTesting = randomUserTesting.sample(n=1,replace=False)
            #random event 
            randomEvent = trainingData.sample(n=1,replace=False)

            #get lat and long of user's training data
            trainingCoordinates=self.get_event_coordinates(randomUserTraining)
            # trainingCoordinates=randomUserTraining.loc[:,['latx', 'longx']]

            #user's coordinates
            userCoordinates=self.get_user_coordinates(userID, randomUserTraining)
            # userCoordinates=randomUserTraining.loc[randomUserTraining.index[0],['laty', 'longy']]
            
            #coordinates of known and unknown event
            knownEventCoordinates=self.get_event_coordinates(randomUserTesting)
            unknownEventCoordinates=self.get_event_coordinates(randomEvent)
            # knownEventCoordinates=randomUserTesting.loc[:,['latx', 'longx']]
            # unknownEventCoordinates=randomEvent.loc[:,['latx', 'longx']]

            #for known event 
            similarityKnown=self.get_score(trainingCoordinates, knownEventCoordinates, userCoordinates)

            # trainingDistancesWithKnownEvent=trainingCoordinates.apply(lambda row: self.get_distance_between_two_points(row['latx'], knownEventCoordinates.latx, row['longx'], knownEventCoordinates.longx), axis=1)
            # distanceFromHomeKnownEvent=self.get_distance_between_two_points(knownEventCoordinates.latx, userCoordinates.laty, knownEventCoordinates.longx, userCoordinates.longy)

            # sigmoidKnown = tf.reduce_mean(input_tensor=tf.sigmoid(-trainingDistancesWithKnownEvent))
            # sigmoidHome = tf.sigmoid(distanceFromHomeKnownEvent)

            # similarityKnown = (sigmoidKnown + sigmoidHome/2)/2



            #for unknown event 
            # trainingDistancesWithUnknownEvent=trainingCoordinates.apply(lambda row: self.get_distance_between_two_points(row['latx'], unknownEventCoordinates.latx, row['longx'], unknownEventCoordinates.longx), axis=1)
            # distanceFromHomeUnknownEvent=self.get_distance_between_two_points(unknownEventCoordinates.latx, userCoordinates.laty, unknownEventCoordinates.longx, userCoordinates.longy)

            # sigmoidUnknown = tf.reduce_mean(input_tensor=tf.sigmoid(-trainingDistancesWithUnknownEvent))
            # sigmoidHomeUnknown = tf.sigmoid(distanceFromHomeUnknownEvent)

            # similarityUnknown = (sigmoidUnknown + sigmoidHomeUnknown/2)/2
            similarityUnknown = self.get_score(trainingCoordinates, unknownEventCoordinates, userCoordinates)

            diff.append((similarityKnown - similarityUnknown) > 0)

            
        diff = np.asarray(diff)
        auc = np.mean(diff == True)
        print(auc)
        return auc

    def get_score(self, prevEventsCoordinates, eventCoordinates, userCoordinates):
        #get the random user
        prevEventsDistancesWithEvent=prevEventsCoordinates.apply(lambda row: self.get_distance_between_two_points(row['latx'], eventCoordinates.latx, row['longx'], eventCoordinates.longx), axis=1)
        distanceFromHome=self.get_distance_between_two_points(eventCoordinates.latx, userCoordinates.laty, eventCoordinates.longx, userCoordinates.longy)

        sigmoidPrevEvents = tf.reduce_mean(input_tensor=tf.sigmoid(-prevEventsDistancesWithEvent))
        sigmoidHome = tf.sigmoid(distanceFromHome)

        similarity = (sigmoidPrevEvents + sigmoidHome/2)/2

        return similarity

    def get_event_coordinates(self, events):

        #get lat and long of user's training data
        trainingCoordinates=events.loc[:,['latx', 'longx']]

        return trainingCoordinates

    def get_user_coordinates(self, userID, prevEvents):
        #find the user
        randomUserTraining = prevEvents[prevEvents.user_id == userID]

        #user's coordinates
        userCoordinates=randomUserTraining.loc[randomUserTraining.index[0],['laty', 'longy']]
        
        return userCoordinates


    def make_recommendation(self, event_dictionary, user_id=None, num_items=10):

        return ''


    def __del__(self):
        tf.compat.v1.reset_default_graph()
        print ("deleted")
