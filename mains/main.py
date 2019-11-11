import tensorflow as tf
import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import sys
sys.path.append(".") # Adds higher directory to python modules path.
from tqdm import tqdm
from data_loader.data_generator import DataGenerator
from base.matrix_factorization_model import MFModel
from figures.diagrams import create_diagram




def main():
    
    # DATA PREPERATION

    dataClass = DataGenerator()
    dataClass.events_rsvp_dataset()
    events, users, puids, peids, ptuids, pteids  = dataClass.events, dataClass.users, dataClass.puids, dataClass.peids, dataClass.ptuids, dataClass.pteids
    pvuids, pveids, nuids, neids, ntuids, nteids, nvuids, nveids = dataClass.pvuids, dataClass.pveids, dataClass.nuids, dataClass.neids, dataClass.ntuids, dataClass.nteids, dataClass.nvuids, dataClass.nveids
    
    #HYPERPARAMS

    epochs = 60
    batches = 40 #batches must be as the number of samples (SGD -> Batch size = 1 ) j_bias +
    num_factors = 10 #latent features

    #lambda regularizations 
    lambda_user = 0.0000001
    lambda_event = 0.0000001
    learning_rate = 0.005

    # The number of (u,i,j) triplets we sample for each batch.
    samples = 15000

    #TENSORFLOW GRAPH

    #Set graph
    graph = tf.Graph()
    
    # Run the session. 
    session = tf.compat.v1.Session(config=None, graph=graph)

    loss_array = []
    auc_array = []
    num_factors_array = []
    for i in range(1,11):
        #init model class
        MF_Model = MFModel(graph, users, events, session)

        #build the model
        MF_Model.build_model(num_factors*i, lambda_user, lambda_event, learning_rate)

        #train the model
        MF_Model.train_model(epochs, batches, puids, peids, nuids, neids, pvuids, pveids, nvuids, nveids, num_factors*i, samples)

        # validate our model
        l, auc = MF_Model.validate_model(ptuids, pteids, ntuids, nteids, samples)
        loss_array.append(l)
        auc_array.append(auc)
        num_factors_array.append(num_factors*i)
        
    create_diagram(num_factors_array, loss_array, auc_array, 'Number of Factors', 'Loss', 'AUC', '', '', 'Loss_AUC_testing.png')



if __name__ == '__main__':
    main()
