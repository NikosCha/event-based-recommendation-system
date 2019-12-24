import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import sys
sys.path.append(".") # Adds higher directory to python modules path.
from tqdm import tqdm
from data_loader.data_generator import DataGenerator
from base.matrix_factorization_model import MFModel
from models.textualSimilarity import TextualModel
from models.spatialModel import SpatialModel
from models.socialModel import SocialModel
from figures.diagrams import create_diagram
import datetime




#matrix factorization model
def main():
    import tensorflow as tf
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
    for i in range(1,2):
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


#semantic model
def main2():
    import tensorflow as tf

    
    # DATA PREPERATION
    dataClass = DataGenerator()
    trainingData, testingData = dataClass.contextual_features('semantic', 'Chicago')    

    graph = tf.Graph()
    TS_Model = TextualModel(graph)

    TS_Model.validate_model(trainingData, testingData, 10000)


#spatial model
def main3():
    import tensorflow as tf
    
    # DATA PREPERATION
    dataClass = DataGenerator()
    trainingData, testingData = dataClass.contextual_features('spatial','Chicago')    

    graph = tf.Graph()
    TS_Model = SpatialModel(graph)

    TS_Model.validate_model(trainingData, testingData, 10000)


#textual and spatial model
def main4():
    import tensorflow as tf
    
    # DATA PREPERATION
    dataClass = DataGenerator()
    trainingData, testingData = dataClass.contextual_features('semAndSpat','Chicago')    

    graph = tf.Graph()
    TX_Model = TextualModel(graph)
    TS_Model = SpatialModel(graph)

    diff = []
    #get unique user's ids 
    #we get testing data because if a user exist
    uniqueUsers = np.unique(testingData.user_id)

    for _ in range(0,10000):
        index =  np.random.randint(len(uniqueUsers), size=1)

        #get the random user
        userID=uniqueUsers[index][0]
        randomUserTraining = trainingData[trainingData.user_id == userID]  
        randomUserTesting = testingData[testingData.user_id == userID]

        #get one random event of user's testing set
        randomUserTesting = randomUserTesting.sample(n=1,replace=False)
        #random event 
        randomEvent = trainingData.sample(n=1,replace=False)

        #------------SPATIAL-----------------
        #get lat and long of user's training data
        trainingCoordinates=TS_Model.get_event_coordinates(randomUserTraining)

        #user's coordinates
        userCoordinates=TS_Model.get_user_coordinates(userID, randomUserTraining)
        
        #coordinates of known and unknown event
        knownEventCoordinates=TS_Model.get_event_coordinates(randomUserTesting)
        unknownEventCoordinates=TS_Model.get_event_coordinates(randomEvent)

        #for known event 
        spatialSimilarityKnown=TS_Model.get_score(trainingCoordinates, knownEventCoordinates, userCoordinates)

        #for unknown event 
        spatialSimilarityUnknown = TS_Model.get_score(trainingCoordinates, unknownEventCoordinates, userCoordinates)

        #------------TEXTUAL---------------------
        #prepare data
        randomUserTraining=TX_Model.prepare_description(randomUserTraining)
        randomUserTesting=TX_Model.prepare_description(randomUserTesting)
        randomEvent=TX_Model.prepare_description(randomEvent)

        user_vector = np.mean(randomUserTraining['description'],0)

        randomUserTesting = randomUserTesting.reset_index()
        randomEvent = randomEvent.reset_index()

        cosKnownEvent=TX_Model.get_cosine_similarity(user_vector,randomUserTesting.loc[0,'description'])
        cosUnknownEvent=TX_Model.get_cosine_similarity(user_vector,randomEvent.loc[0,'description'])

        diff.append((cosKnownEvent.numpy()[0]*spatialSimilarityKnown - cosUnknownEvent.numpy()[0]*spatialSimilarityUnknown) > 0)

    diff = np.asarray(diff)
    auc = np.mean(diff == True)
    print('----SPATIAL AND SEMANTIC COMPINATION VALIDATION----')
    print(auc)

#social model
def main5():
    import tensorflow as tf
    
    # DATA PREPERATION
    dataClass = DataGenerator()
    trainingData, testingData = dataClass.contextual_features('social','San Jose')
    users_groups = pd.read_csv('/home/nikoscha/Documents/ThesisR/datasets/group_users.csv', names=['group_id','user'])

    user_dictionary, event_dictionary = dataClass.user_dictionary, dataClass.event_dictionary 
    
    #delete unwanted row of columns
    users_groups = users_groups.drop(users_groups.index[0])    
    users_groups = pd.merge(users_groups, user_dictionary, on='user')
    users_groups = users_groups.astype('int32')
    graph = tf.Graph()
    S_Model = SocialModel(graph, users_groups, user_dictionary, event_dictionary)

    S_Model.validate_model(trainingData, testingData, 10000)


#textual, spatial and social model
def main6():
    import tensorflow as tf
    
    # DATA PREPERATION
    dataClass = DataGenerator()
    trainingData, testingData = dataClass.contextual_features('semSpatSoc','Chicago')    

    users_groups = pd.read_csv('/home/nikoscha/Documents/ThesisR/datasets/group_users.csv', names=['group_id','user'])
    user_dictionary, event_dictionary = dataClass.user_dictionary, dataClass.event_dictionary 
    #delete unwanted row of columns
    users_groups = users_groups.drop(users_groups.index[0])    
    users_groups = pd.merge(users_groups, user_dictionary, on='user')
    users_groups = users_groups.astype('int32')

    graph = tf.Graph()
    TX_Model = TextualModel(graph)
    TS_Model = SpatialModel(graph)
    S_Model = SocialModel(graph, users_groups, user_dictionary, event_dictionary)

    diff = []
    #get unique user's ids 
    #we get testing data because if a user exist
    uniqueUsers = np.unique(testingData.user_id)

    for _ in range(0,1000):
        index =  np.random.randint(len(uniqueUsers), size=1)

        #get the random user
        userID=uniqueUsers[index][0]            
        
        #get his groups
        user_groups = S_Model.get_user_groups(userID)
        #get his friends (people from the same group) 
        user_friends = S_Model.get_user_friend_list(user_groups,userID)

        randomUserTraining = trainingData[trainingData.user_id == userID]  
        randomUserTesting = testingData[testingData.user_id == userID]

        #get one random event of user's testing set
        randomUserTesting = randomUserTesting.sample(n=1,replace=False)
        #random event 
        randomEvent = trainingData.sample(n=1,replace=False)

        #------------SPATIAL-----------------
        #get lat and long of user's training data
        trainingCoordinates=TS_Model.get_event_coordinates(randomUserTraining)

        #user's coordinates
        userCoordinates=TS_Model.get_user_coordinates(userID, randomUserTraining)
        
        #coordinates of known and unknown event
        knownEventCoordinates=TS_Model.get_event_coordinates(randomUserTesting)
        unknownEventCoordinates=TS_Model.get_event_coordinates(randomEvent)

        #for known event 
        spatialSimilarityKnown=TS_Model.get_score(trainingCoordinates, knownEventCoordinates, userCoordinates)

        #for unknown event 
        spatialSimilarityUnknown = TS_Model.get_score(trainingCoordinates, unknownEventCoordinates, userCoordinates)

        #------------TEXTUAL---------------------
        #prepare data
        randomUserTraining=TX_Model.prepare_description(randomUserTraining)
        randomUserTesting=TX_Model.prepare_description(randomUserTesting)
        randomEvent=TX_Model.prepare_description(randomEvent)

        user_vector = np.mean(randomUserTraining['description'],0)

        randomUserTesting = randomUserTesting.reset_index()
        randomEvent = randomEvent.reset_index()

        cosKnownEvent=TX_Model.get_cosine_similarity(user_vector,randomUserTesting.loc[0,'description'])
        cosUnknownEvent=TX_Model.get_cosine_similarity(user_vector,randomEvent.loc[0,'description'])

        #--------------SOCIAL------------------------
        socialScoreKnown = 1
        socialScoreUnknown = 1  
        if len(user_friends) != 0:  #if user doesnt have friends dont use the model
            #get total number of rsvps    
            totalNumberOfRsvps_known = S_Model.get_number_of_rsvps(randomUserTesting.event_id,trainingData)
            #get total friends who will attend event
            totalFriendsRsvps_known = S_Model.get_friends_rsvps_from_events(user_friends,randomUserTesting.event_id,trainingData)

            #same for unknow event  
            totalNumberOfRsvps_unknown = S_Model.get_number_of_rsvps(randomEvent.event_id,trainingData)
            totalFriendsRsvps_unknown = S_Model.get_friends_rsvps_from_events(user_friends,randomEvent.event_id,trainingData)

            socialScoreKnown = S_Model.get_score(totalNumberOfRsvps_known, totalFriendsRsvps_known)
            socialScoreUnknown = S_Model.get_score(totalNumberOfRsvps_unknown, totalFriendsRsvps_unknown) 

        

        diff.append((cosKnownEvent.numpy()[0]*spatialSimilarityKnown*socialScoreKnown - cosUnknownEvent.numpy()[0]*spatialSimilarityUnknown*socialScoreUnknown) > 0)

    diff = np.asarray(diff)
    auc = np.mean(diff == True)
    print('----SPATIAL, SEMANTIC and SOCIAL COMPINATION VALIDATION----')
    print(auc)


if __name__ == '__main__':
    # main()
    # main2()
    # main3()
    # main4()
    # main5()
    main6()