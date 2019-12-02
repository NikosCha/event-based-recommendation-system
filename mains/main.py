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
import datetime





def main():
    try:
        import tensorflow.compat.v2 as tf
    except Exception:
        pass

    tf.enable_v2_behavior()

    print(tf.__version__)
    # DATA PREPERATION

    dataClass = DataGenerator()
    dataClass.events_rsvp_dataset()
    events, users, puids, peids, ptuids, pteids  = dataClass.events, dataClass.users, dataClass.puids, dataClass.peids, dataClass.ptuids, dataClass.pteids
    pvuids, pveids, nuids, neids, ntuids, nteids, nvuids, nveids = dataClass.pvuids, dataClass.pveids, dataClass.nuids, dataClass.neids, dataClass.ntuids, dataClass.nteids, dataClass.nvuids, dataClass.nveids
    

    event_dictionary, user_dictionary = dataClass.event_dictionary, dataClass.user_dictionary

    #HYPERPARAMS

    epochs = 3
    batches = 4 #batches must be as the number of samples (SGD -> Batch size = 1 ) j_bias +
    num_factors = 10 #latent features

    #lambda regularizations 
    lambda_user = 0.0000001
    lambda_event = 0.0000001
    learning_rate = 0.001

    # The number of (u,i,j) triplets we sample for each batch.
    samples = 15000

    #TENSORFLOW GRAPH

    #Set graph
    graph = tf.Graph()

    loss_array = []
    auc_array = []
    num_factors_array = []
    time_array = []
    for i in range(1,3):
        #init model class
        MF_Model = MFModel(graph, users, events)

        #build the model
        MF_Model.build_model(num_factors*i, lambda_user, lambda_event, 0.005)

        #get start time
        start = datetime.datetime.now().timestamp()

        #train the model
        MF_Model.train_model(epochs, batches, puids, peids, nuids, neids, pvuids, pveids, nvuids, nveids, num_factors*i, samples, 0.005)

        #get end time
        end = datetime.datetime.now().timestamp()
        trainTime = end - start
        time_array.append(trainTime)

        # validate our model
        l, auc = MF_Model.validate_model(ptuids, pteids, ntuids, nteids, samples)

        #get recommendations 
        recommendation = MF_Model.make_recommendation(event_dictionary, user_id=1)
        print(recommendation)

        
        # reset variables and session
        del MF_Model

        loss_array.append(l)
        auc_array.append(auc)
        num_factors_array.append(num_factors*i)
        
    create_diagram(num_factors_array, loss_array, auc_array, 'Number of Factors', 'Loss', 'AUC', '', '', 'Loss_AUC_testing.png', 2)
    create_diagram(num_factors_array, time_array, '', 'Number of Epochs', 'Time (s)', '', '', '', 'Time_Test.png', 1)


    #tests for learning rate
    loss_array = []
    auc_array = []
    learning_rate_array = []
    time_array = []
    for i in range(1,11):
        #init model class
        MF_Model = MFModel(graph, users, events)

        #build the model
        MF_Model.build_model(40, lambda_user, lambda_event, learning_rate*i)

        #get start time
        start = datetime.datetime.now().timestamp()

        #train the model
        MF_Model.train_model(epochs, batches, puids, peids, nuids, neids, pvuids, pveids, nvuids, nveids, 40, samples, learning_rate*i)

        #get end time
        end = datetime.datetime.now().timestamp()
        trainTime = end - start
        time_array.append(trainTime)
        
        # validate our model
        l, auc = MF_Model.validate_model(ptuids, pteids, ntuids, nteids, samples)

        # reset variables and session
        del MF_Model

        loss_array.append(l)
        auc_array.append(auc)
        learning_rate_array.append(learning_rate*i)
        
    create_diagram(learning_rate_array, loss_array, auc_array, 'Learning Rate', 'Loss', 'AUC', '', '', 'Loss_AUC_LR.png', 2)
    create_diagram(learning_rate_array, time_array, '', 'Learning Rate', 'Time (s)', '', '', '', 'Time_Test_LR.png', 1)




if __name__ == '__main__':
    main()
