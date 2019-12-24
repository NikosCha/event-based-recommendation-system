import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from figures.diagrams import create_diagram
from utils.variables import get_variable, init_variable


class SocialModel:
    def __init__(self, graph, users_groups, user_dictionary, event_dictionary):
        self.graph = graph
        self.users_groups = users_groups
        self.user_dictionary = user_dictionary
        self.event_dictionary = event_dictionary
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
            
            #get his groups
            user_groups = self.get_user_groups(userID)
            #get his friends (people from the same group) 
            user_friends = self.get_user_friend_list(user_groups,userID)

            if len(user_friends) == 0:  
                continue

            randomUserTesting = testingData[testingData.user_id == userID]
            #get one random event of user's testing set
            randomUserTesting = randomUserTesting.sample(n=1,replace=False)
            #random event 
            randomEvent = trainingData.sample(n=1,replace=False)

            #get total number of rsvps    
            totalNumberOfRsvps_known = self.get_number_of_rsvps(randomUserTesting.event_id,trainingData)
            #get total friends who will attend event
            totalFriendsRsvps_known = self.get_friends_rsvps_from_events(user_friends,randomUserTesting.event_id,trainingData)

            #same for unknow event  
            totalNumberOfRsvps_unknown = self.get_number_of_rsvps(randomEvent.event_id,trainingData)
            totalFriendsRsvps_unknown = self.get_friends_rsvps_from_events(user_friends,randomEvent.event_id,trainingData)

            scoreKnown = self.get_score(totalNumberOfRsvps_known, totalFriendsRsvps_known)
            scoreUnknown = self.get_score(totalNumberOfRsvps_unknown, totalFriendsRsvps_unknown)

            diff.append((scoreKnown - scoreUnknown) > 0)

            
        diff = np.asarray(diff)
        auc = np.mean(diff == True)
        print('----SOCIAL MODEL VALIDATION----')
        print(auc)
        return auc

    def get_score(self, rsvpsNumber, friendsRsvpNumber):
        score = self.sigmoid(friendsRsvpNumber)
        return score

    def sigmoid(self,x):
        return 1 / (1 + math.exp(-x))

    #from dictionary_id get real_id
    def get_user_real_id(self, dictionary_id):
        real_id = self.user_dictionary.user.loc[self.user_dictionary.user_id == str(dictionary_id)].iloc[0]
        return real_id
    
    #from dictionary_id get real_id
    def get_event_real_id(self, dictionary_id):
        real_id = self.event_dictionary.event.loc[self.event_dictionary.event_id == str(dictionary_id)].iloc[0]
        return real_id

    #from real_id get dictionary_id 
    def get_user_dictionary_id(self, real_id):
        try:
            dictionary_id = self.user_dictionary.user_id.loc[self.user_dictionary.user == int(real_id)].iloc[0]
            return dictionary_id
        except IndexError as error: #it means user doesnt not exist in rsvps data
            return -1

    #from real_id get dictionary_id
    def get_event_dictionary_id(self, real_id):
        dictionary_id = self.event_dictionary.event_id.loc[self.event_dictionary.event == int(real_id)].iloc[0]
        return dictionary_id

    # get -1 if there are no friends 
    def get_user_friend_list(self, groups, user_id):
        friend_list = []
        for group in groups:
            users_official_ids = self.users_groups.user_id.loc[((self.users_groups.group_id) == int(group)) & (self.users_groups.user_id != int(user_id))]
            friend_list.extend(users_official_ids)
        return friend_list

    def get_user_groups(self, user_id):
        groups = self.users_groups.group_id.loc[self.users_groups.user_id == int(user_id)]
        groups = groups.values
        return groups

    def get_number_of_rsvps(self, event_id, rsvps):
        nOfRsvps = rsvps.user_id.loc[rsvps.event_id == int(event_id)]
        return len(nOfRsvps)

    def get_friends_rsvps_from_events(self, friends_ids, event_id, rsvps):
        rsvps = rsvps.user_id.loc[rsvps.event_id == int(event_id)].values
        friendsInEventRsvps = np.in1d(rsvps, friends_ids)                
        return len(friendsInEventRsvps[friendsInEventRsvps == True])

    def make_recommendation(self, event_dictionary, user_id=None, num_items=10):

        return ''    

    def __del__(self):
        tf.compat.v1.reset_default_graph()
        print ("deleted")
