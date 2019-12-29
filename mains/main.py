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
from models.temporalModel import TemporalModel
from figures.diagrams import create_diagram
import datetime
import math




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

    epochs = 60
    batches = 100 #batches must be as the number of samples (SGD -> Batch size = 1 ) j_bias +
    num_factors = 60 #latent features

    #lambda regularizations 
    lambda_user = 0.0000001
    lambda_event = 0.0000001
    learning_rate = 0.001

    # The number of (u,i,j) triplets we sample for each batch.
    samples = 5000

    #TENSORFLOW GRAPH

    #Set graph
    graph = tf.Graph()

    loss_array = []
    auc_array = []
    num_factors_array = []
    time_array = []
    for i in range(1,11):
        tf.compat.v1.reset_default_graph()
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

        # #get recommendations 
        # recommendation = MF_Model.make_recommendation(event_dictionary, user_id=1)
        # print(recommendation)

        
        # reset variables and session
        del MF_Model

        loss_array.append(l)
        auc_array.append(auc)
        num_factors_array.append(num_factors*i)
        
    create_diagram(num_factors_array, loss_array, auc_array, 'Number of Factors', 'Loss', 'AUC', '', '', 'Loss_AUC_testing.png', 2)
    create_diagram(num_factors_array, time_array, '', 'Number of Factors', 'Time (s)', '', '', '', 'Time_Test.png', 1)


    #tests for learning rate
    loss_array = []
    auc_array = []
    learning_rate_array = []
    time_array = []
    for i in range(1,7):
        tf.compat.v1.reset_default_graph()
        #init model class
        MF_Model = MFModel(graph, users, events)

        #build the model
        MF_Model.build_model(60, lambda_user, lambda_event, learning_rate*i)

        #get start time
        start = datetime.datetime.now().timestamp()

        #train the model
        MF_Model.train_model(epochs, batches, puids, peids, nuids, neids, pvuids, pveids, nvuids, nveids, 60, samples, learning_rate*i)

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
    print('----SPATIAL AND SEMANTIC COMBINATION VALIDATION----')
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
    print('----SPATIAL, SEMANTIC and SOCIAL COMBINATION VALIDATION----')
    print(auc)

#temporal model
def main7():
    import tensorflow as tf
    
    # DATA PREPERATION
    dataClass = DataGenerator()
    trainingData, testingData = dataClass.contextual_features('temporal','Chicago')
    
    graph = tf.Graph()
    TM_Model = TemporalModel(graph)

    TM_Model.validate_model(trainingData, testingData, 10000)

#textual, spatial, social model and temporal
def main8():
    import tensorflow as tf
    
    # DATA PREPERATION
    dataClass = DataGenerator()
    trainingData, testingData = dataClass.contextual_features('semSpatSocTemp','Phoenix')    

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
    TM_Model = TemporalModel(graph)

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

        #------------TEMPORAL-----------------------

        userVector = TM_Model.get_user_vector(randomUserTraining)
        eventVectorKnown = TM_Model.get_event_vector(randomUserTesting)
        eventVectorUnknown = TM_Model.get_event_vector(randomEvent)

        temporalSimilarityKnown = TM_Model.get_cosine_similarity(userVector, eventVectorKnown)
        temporalSimilarityUnknown = TM_Model.get_cosine_similarity(userVector, eventVectorUnknown)

        diff.append((cosKnownEvent.numpy()[0]*spatialSimilarityKnown*socialScoreKnown*temporalSimilarityKnown - cosUnknownEvent.numpy()[0]*spatialSimilarityUnknown*socialScoreUnknown*temporalSimilarityUnknown) > 0)

    diff = np.asarray(diff)
    auc = np.mean(diff == True)
    print('----SPATIAL, SEMANTIC, SOCIAL AND TEMPORAL COMBINATION VALIDATION----')
    print(auc)

#both models
def main9(city):
    import tensorflow as tf
    
    # DATA PREPERATION
    dataClass = DataGenerator()
    print(city)
    dataClass.general_data(city)
    trainingData, testingData = dataClass.dfTraining, dataClass.dfTestingAndValidation   
    events, users, puids, peids, ptuids, pteids  = dataClass.events, dataClass.users, dataClass.puids, dataClass.peids, dataClass.ptuids, dataClass.pteids
    

    users_groups = pd.read_csv('/home/nikoscha/Documents/ThesisR/datasets/group_users.csv', names=['group_id','user'])
    user_dictionary, event_dictionary = dataClass.user_dictionary, dataClass.event_dictionary 
    #delete unwanted row of columns
    users_groups = users_groups.drop(users_groups.index[0])    
    users_groups = pd.merge(users_groups, user_dictionary, on='user')
    users_groups = users_groups.astype('int32')

    graph = tf.Graph()

    MF_Model = MFModel(graph, users, events)
    #build the model
    MF_Model.build_model(80, 0.0000001, 0.0000001, 0.001)
    #train the model
    MF_Model.train_model_without_validation(100, 100, puids, peids, '', '', 80, 15000, 0.001)
   

    TX_Model = TextualModel(graph)
    TS_Model = SpatialModel(graph)
    S_Model = SocialModel(graph, users_groups, user_dictionary, event_dictionary)
    TM_Model = TemporalModel(graph)

    diffMF = []
    diffCont = []
    diffFinal = []
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

        event_id_known = randomUserTesting.event_id.values
        len_event_known = len(trainingData[trainingData.event_id == event_id_known[0]])            

        #random event 
        randomEvent = trainingData.sample(n=1,replace=False)
        
        event_id_unknown = randomEvent.event_id.values
        len_event_unknown = len(trainingData[trainingData.event_id == event_id_unknown[0]])

        #MATRIX FACTORIZATION -- COLLABORATIVE FILTERING MODEL
        MFScoreKnown = MF_Model.get_score(userID,randomUserTesting.event_id)
        MFScoreUnknown = MF_Model.get_score(userID,randomEvent.event_id)

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

        #------------TEMPORAL-----------------------

        userVector = TM_Model.get_user_vector(randomUserTraining)
        eventVectorKnown = TM_Model.get_event_vector(randomUserTesting)
        eventVectorUnknown = TM_Model.get_event_vector(randomEvent)

        temporalSimilarityKnown = TM_Model.get_cosine_similarity(userVector, eventVectorKnown)
        temporalSimilarityUnknown = TM_Model.get_cosine_similarity(userVector, eventVectorUnknown)

        #------------- FINAL MODEL --------------------
        contextualFeaturesKnown = cosKnownEvent.numpy()[0]*spatialSimilarityKnown*socialScoreKnown*temporalSimilarityKnown
        contextualFeaturesUnknown = cosUnknownEvent.numpy()[0]*spatialSimilarityUnknown*socialScoreUnknown*temporalSimilarityUnknown

        #linear combination 
        # if MFScoreKnown and MFScoreUnknown are negative, make them positive

        a_known = 1/(1 + np.log(len_event_known + 1))
        a_unknown = 1/(1 + np.log(len_event_unknown + 1))

        # if (MFScoreUnknown < 0 and MFScoreKnown < 0): 
        #     temp = MFScoreKnown
        #     MFScoreKnown = -MFScoreUnknown
        #     MFScoreUnknown = -temp
        # if MFScoreKnown < 0 :
        #     MFScoreKnown = abs(MFScoreKnown) # prepei na einai to distance metaksi known kai unknown an einai ena agnwsto
        #     MFScoreUnknown = abs(MFScoreUnknown) # prepei na einai to distance metaksi known kai unknown an einai ena agnwsto

        #sigmoid to MF to get range from 0 to 1 
        MFScoreKnown = 1 / (1 + math.exp(-MFScoreKnown))
        MFScoreUnknown = 1 / (1 + math.exp(-MFScoreUnknown))


        diffFinal.append((((a_known*contextualFeaturesKnown)*((1 - a_known)*(MFScoreKnown))) - ( (a_unknown*contextualFeaturesUnknown)*((1-a_unknown)*(MFScoreUnknown))) ) > 0)
        diffMF.append((MFScoreKnown - MFScoreUnknown) > 0)
        diffCont.append((contextualFeaturesKnown - contextualFeaturesUnknown) > 0)


    diffFinal = np.asarray(diffFinal)
    aucFinal = np.mean(diffFinal == True)
    print('----FINAL MF AND CONT----')
    print(aucFinal)

    diffMF = np.asarray(diffMF)
    aucMF = np.mean(diffMF == True)
    print('----MF----')
    print(aucMF)

    diffCont = np.asarray(diffCont)
    aucCont = np.mean(diffCont == True)
    print('----CONT----')
    print(aucCont)
    

if __name__ == '__main__':
    main()
    # main2()
    # main3()
    # main4()
    # main5()
    # main6()
    # main7()
    # main8()
    # main9('Chicago')
    # main9('Phoenix')
    # main9('San Jose')