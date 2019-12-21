import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from bs4 import BeautifulSoup
from figures.diagrams import create_diagram
from utils.variables import get_variable, init_variable
import tensorflow_hub as hub


class TextualModel:
    def __init__(self, graph):

        hub_url = 'https://tfhub.dev/google/universal-sentence-encoder/4'
        self.embed = hub.KerasLayer(hub_url)
        self.graph = graph
        self.session = tf.compat.v1.Session(config=None, graph=graph)

    def cleanhtml(self, html): 
        soup = BeautifulSoup(html, features="html.parser")
        return soup.get_text()

    def get_cosine_similarity(self, x1, x2, name='Cosine_loss'):
        with tf.name_scope(name):
            x1_val = tf.sqrt(tf.reduce_sum(tf.matmul(x1,tf.transpose(x1)),axis=1))
            x2_val = tf.sqrt(tf.reduce_sum(tf.matmul(x2,tf.transpose(x2)),axis=1))
            denom =  tf.multiply(x1_val,x2_val)
            num = tf.reduce_sum(tf.multiply(x1,x2),axis=1)
            return tf.compat.v1.div(num,denom)
    
    def get_sentence_embeddings(self, sentence):
        sentence = self.embed([sentence])
        return sentence.numpy()

    def prepare_description(self, df):
        #clean html tags etc from descriptions
        df.loc[:,'description'] = df.apply(lambda row: self.cleanhtml(row.description), axis=1)
        #set description as 512 vector(embedding) 
        df.loc[:,'description'] = df.apply(lambda row: self.get_sentence_embeddings(row.description), axis=1)

        return df

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
            #random event for each user testing evnet
            randomEvents = trainingData.sample(n=1,replace=False)

            #prepare data
            randomUserTraining=self.prepare_description(randomUserTraining)
            randomUserTesting=self.prepare_description(randomUserTesting)
            randomEvents=self.prepare_description(randomEvents)

            user_vector = np.mean(randomUserTraining['description'],0)

            randomUserTesting = randomUserTesting.reset_index()
            randomEvents = randomEvents.reset_index()
        
            cos1=self.get_cosine_similarity(user_vector,randomUserTesting.loc[0,'description'])
            cos2=self.get_cosine_similarity(user_vector,randomEvents.loc[0,'description'])

            diff.append((cos1.numpy()[0] - cos2.numpy()[0]) > 0)
            
        diff = np.asarray(diff)
        auc = np.mean(diff == True)
        print('----SEMANTIC MODEL VALIDATION----')
        print(auc)
        return auc

    def make_recommendation(self, event_dictionary, user_id=None, num_items=10):
        """Recommend items for a given user given a trained model
        Args:
            user_id (int): The id of the user we want to create recommendations for.
            num_items (int): How many recommendations we want to return.
        Returns:
            recommendations (pandas.DataFrame): DataFrame with num_items artist names and scores
        """

        # Grab our user matrix U
        # user_vecs = get_variable(self.graph, self.session, 'user_factors')
        with self.graph.as_default():
            user_vecs = self.user_factors.eval(session=self.session)


            # Grab our item matrix V
            # item_vecs = get_variable(self.graph, self.session, 'event_factors')
            item_vecs = self.event_factors.eval(session=self.session)

        # Calculate the score for our user for all items. 
        rec_vector = user_vecs[user_id, :].dot(item_vecs.T)

        # Grab the indices of the top users
        item_idx = np.argsort(rec_vector)[::-1][:num_items]

        # Map the indices to event names and add to dataframe along with scores.
        events, scores = [], []

        for idx in item_idx:
            events.append(event_dictionary.event.loc[event_dictionary.event_id == str(idx)].iloc[0])
            scores.append(rec_vector[idx])

        recommendations = pd.DataFrame({'events': events, 'score': scores})

        return recommendations


    def __del__(self):
        tf.compat.v1.reset_default_graph()
        print ("deleted")
