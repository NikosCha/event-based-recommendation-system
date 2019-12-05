import datetime
import numpy as np
import pandas as pd
import tensorflow.compat.v2 as tf
from tqdm import tqdm
from bs4 import BeautifulSoup
from figures.diagrams import create_diagram
from utils.variables import get_variable, init_variable
import tensorflow_hub as hub

tf.enable_v2_behavior()
class TextualModel:
    def __init__(self, graph):

        hub_url = '/tmp/module/universal_module/'
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
    
    def get_sentence_embedding(self, sentence):
        sentence = self.embed([sentence])
        return sentence['outputs'].numpy()

            #clean html tags etc from descriptions
            randomUserTraining.loc[:,'description'] = randomUserTraining.apply(lambda row: self.cleanhtml(row.description), axis=1)
            randomUserTesting.loc[:,'description'] = randomUserTesting.apply(lambda row: self.cleanhtml(row.description), axis=1)
            randomEvents.loc[:,'description'] = randomEvents.apply(lambda row: self.cleanhtml(row.description), axis=1)

