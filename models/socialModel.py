import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from bs4 import BeautifulSoup
from figures.diagrams import create_diagram
from utils.variables import get_variable, init_variable


class SocialModel:
    def __init__(self, graph, users_groups):
        self.graph = graph
        self.users_groups = users_groups
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

        diff = np.asarray(diff)
        auc = np.mean(diff == True)
        print('----SOCIAL MODEL VALIDATION----')
        print(auc)
        return auc


    def make_recommendation(self, event_dictionary, user_id=None, num_items=10):

        return ''

    def get_user_friend_list(self, user_id, groups):

        return 'user friends'

    def get_user_groups(self, user_id):
        
        return 'user groups'

    def get_number_of_rsvps(self, event_id):
       
        return 'number of rsvps'

    def get_friends_rsvps_from_events(self, friends_ids, event_id):

        return 'number of friends in rsvps'

    def __del__(self):
        tf.compat.v1.reset_default_graph()
        print ("deleted")
