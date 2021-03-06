import numpy as np
import pandas as pd
import scipy.sparse as sp
from bs4 import BeautifulSoup


class DataGenerator:
    def __init__(self):
        self.event_dictionary = ''
        self.user_dictionary = ''
        self.events = ''
        self.users = ''
        self.uids = '' 
        self.eids = ''
        self.utids = ''
        self.etids = ''

    def events_rsvp_dataset(self,city):

        #read the dataset
        df = pd.read_csv('/home/nikoscha/Documents/ThesisR/datasets/dataset_new_new.csv', names=['user','response_nn', 'time', 'utc_offset','event','created','description', 'group_id','latx', 'longx','city' ,'laty', 'longy', 'distance', 'weekday'])
        #response is 
        # 0 for yes
        # 1 for no
        # 2 for waitlist

        #delete unwanted row of columns
        df = df.drop(df.index[0])

        #drop cities we dont want
        if city != 'all' :
            df = df[df.city == city]

        df = df.drop(['time', 'utc_offset', 'description', 'group_id','latx', 'longx','city' ,'laty', 'longy', 'distance', 'weekday'], axis=1)
        #response to int because response has string and int as dtypes
        df['responses'] = df['response_nn'].astype("int8")

        # waitlist is approximately 1.3 % so I will delete it 
        # df = df.loc[df.responses != 2]
        df = df[df.responses != 2]

        #delete negative responses
        df = df[df.responses != 1]

        # set the no response as number -1 instead of 1.
        # df['responses'] = df.responses.replace(1, -1)

        # set the yes response as number 1 instead of 0.
        df['responses'] = df.responses.replace(0, 1)


        #create numerical ids
        df['event_id'] = df['event'].astype("category").cat.codes
        df['user_id'] = df['user'].astype("category").cat.codes    
        df = df.sort_values(by=['event_id', 'created'])
        df = df.reset_index() 


            
        # Create a dictionary so we can get the original ids
        self.event_dictionary = df[['event_id', 'event']].drop_duplicates()
        self.event_dictionary['event_id'] = self.event_dictionary.event_id.astype(str)

        self.user_dictionary = df[['user_id','user']].drop_duplicates()
        self.user_dictionary['user_id'] = self.user_dictionary.user_id.astype(str)

        #drop the old columns 
        df = df.drop(['event', 'user', 'response_nn', 'index'], axis=1)

        #analyze the data

        uniqueUsers, countsUsers = np.unique(df.user_id, return_counts=True)
        #dif frequency values of counts
        uniqueFreqValUsers, uniqueFreqValUsersCounts = np.unique(countsUsers, return_counts=True)
        #add a column with the sum of user RSVPs
        df['userRSVP'] = countsUsers[df.user_id]

        uniqueEvents, countsEvents = np.unique(df.event_id, return_counts=True)
        #dif frequency values of counts
        uniqueFreqValEvents, uniqueFreqValEventsCounts = np.unique(countsEvents, return_counts=True)
        
        #add a column with the sum of event RSVPs
        df['eventRSVP'] = countsEvents[df.event_id]
        df['rank'] = df.groupby(['event_id']).cumcount()+1


        #get 50% of the dataset as training data (25% testing and 25%)
        #give us random values in [0,1] and if <0.5 return true , else false
        mask = round(df['eventRSVP']*0.6).astype('int64') >= df['rank']
        
        dfTraining = df[mask]
        dfTestingAndValidation = df[~mask]
        #split testing to testing and validation
        mask = np.random.rand(len(dfTestingAndValidation)) < 0.5
        dfTesting = dfTestingAndValidation[mask]
        dfValidation = dfTestingAndValidation[~mask]

        #true if user and event exists in training set, false otherwise
        mask = np.logical_and(np.isin(dfTesting.user_id, dfTraining.user_id), np.isin(dfTesting.event_id, dfTraining.event_id))
        dfTesting = dfTesting[mask]
        #same for validation set
        mask = np.logical_and(np.isin(dfValidation.user_id, dfTraining.user_id), np.isin(dfValidation.event_id, dfTraining.event_id))
        dfValidation = dfValidation[mask]

        dfTraining = dfTraining.reset_index() 
        dfValidation = dfValidation.reset_index() 
        dfTesting = dfTesting.reset_index() 

        dfTraining = dfTraining.drop(['index'], axis=1)
        dfValidation = dfValidation.drop(['index'], axis=1)
        dfTesting = dfTesting.drop(['index'], axis=1)



        # Create lists of all events, users and respones
        self.events = list(np.sort(df.event_id.unique()))
        self.users = list(np.sort(df.user_id.unique()))      
        responses = list(dfTraining.responses)
        responsesTesting = list(dfTesting.responses)
        responsesValidation = list(dfValidation.responses)

        # Get the rows and columns for our new matrix
        rows = dfTraining.user_id
        cols = dfTraining.event_id

        rowsTesting = dfTesting.user_id
        colsTesting = dfTesting.event_id

        rowsValidation = dfValidation.user_id
        colsValidation = dfValidation.event_id

        # Contruct a sparse matrix for our users and events containing RSVPs 
        data_sparse = sp.csr_matrix((responses, (rows, cols)), shape=(len(self.users), len(self.events)))
        data_sparse_testing = sp.csr_matrix((responsesTesting, (rowsTesting, colsTesting)), shape=(len(self.users), len(self.events)))
        data_sparse_validation = sp.csr_matrix((responsesValidation, (rowsValidation, colsValidation)), shape=(len(self.users), len(self.events)))

        #get positive feedbacks
        # self.uids, self.eids = data_sparse.nonzero()
        self.puids, self.peids, values = sp.find(data_sparse == 1)

        # self.utids, self.etids = data_sparse_testing.nonzero()
        self.ptuids, self.pteids, valuesTesting = sp.find(data_sparse_testing == 1)

        # self.uvids, self.evids = data_sparse_validation.nonzero()
        self.pvuids, self.pveids, validation = sp.find(data_sparse_validation == 1)


        #get negative feedbacks 
        # self.uids, self.eids = data_sparse.nonzero()
        self.nuids, self.neids, values = sp.find(data_sparse == -1)

        # self.utids, self.etids = data_sparse_testing.nonzero()
        self.ntuids, self.nteids, valuesTesting = sp.find(data_sparse_testing == -1)

        # self.uvids, self.evids = data_sparse_validation.nonzero()
        self.nvuids, self.nveids, validation = sp.find(data_sparse_validation == -1)

    #type is 1) semantic 2) spatial 3) temporal   
    #city can be Chicago, Phoenix, San Jose, Mountain View, Scottsdale etc etc.
    def contextual_features(self, dataType, city): 
        def converter(instr):
            return np.fromstring(instr[1:-1],sep=' ')
        
        #read the dataset
        if city == 'Chicago':
            df = pd.read_csv('/home/nikoscha/Documents/ThesisR/datasets/dataset_with_emb_Chicago.csv' ,converters={'description':converter}, names=['user','response_nn', 'time', 'utc_offset','event','created','description', 'group_id','latx', 'longx','city' ,'laty', 'longy', 'distance', 'weekday'])
        elif city == 'Phoenix':
            df = pd.read_csv('/home/nikoscha/Documents/ThesisR/datasets/dataset_with_emb_Phoenix.csv',converters={'description':converter} , names=['user','response_nn', 'time', 'utc_offset','event','created','description', 'group_id','latx', 'longx','city' ,'laty', 'longy', 'distance', 'weekday'])
        elif city == 'San Jose':
            df = pd.read_csv('/home/nikoscha/Documents/ThesisR/datasets/dataset_with_emb_San Jose.csv',converters={'description':converter} , names=['user','response_nn', 'time', 'utc_offset','event','created','description', 'group_id','latx', 'longx','city' ,'laty', 'longy', 'distance', 'weekday'])

        #delete unwanted row of columns
        df = df.drop(df.index[0])
        

        # df = pd.read_csv('/home/nikoscha/Documents/ThesisR/datasets/dataset_new_new.csv', names=['user','response_nn', 'time', 'utc_offset','event','created','description', 'group_id','latx', 'longx','city' ,'laty', 'longy', 'distance', 'weekday'])
        # #response is 
        # # 0 for yes
        # # 1 for no
        # # 2 for waitlist

        # #delete unwanted row of columns
        # df = df.drop(df.index[0])

        # #drop cities we dont want
        # if city != 'all' :
        #     df = df[df.city == city]

        #response to int because response has string and int as dtypes
        df['responses'] = df['response_nn'].astype("int8")

        # waitlist is approximately 1.3 % so I will delete it 
        # df = df.loc[df.responses != 2]
        df = df[df.responses != 2]

        #delete no response, we just need 
        df = df[df.responses != 1]

        # set the yes response as number 1 instead of 0.
        df['responses'] = df.responses.replace(0, 1)

        #create numerical ids
        df['event_id'] = df['event'].astype("category").cat.codes
        df['user_id'] = df['user'].astype("category").cat.codes    
        df = df.sort_values(by=['event_id', 'created'])
        df = df.reset_index() 


            
        # Create a dictionary so we can get the original ids
        self.event_dictionary = df[['event_id', 'event']].drop_duplicates()
        self.event_dictionary['event_id'] = self.event_dictionary.event_id.astype(str)

        self.user_dictionary = df[['user_id','user']].drop_duplicates()
        self.user_dictionary['user_id'] = self.user_dictionary.user_id.astype(str)

        if dataType == 'semantic':
            #drop unneeded columns
            df = df.drop(['time', 'utc_offset', 'responses', 'group_id','latx', 'longx', 'laty', 'longy', 'city', 'distance', 'weekday'], axis=1)
        elif dataType == 'spatial':
            #drop unneeded columns
            df = df.drop(['time', 'utc_offset', 'responses', 'group_id','description', 'city', 'distance', 'weekday'], axis=1)
        elif dataType == 'semAndSpat': #for semantic and spatial 
            df = df.drop(['time', 'utc_offset', 'responses', 'group_id', 'city', 'distance', 'weekday'], axis=1)
        elif dataType == 'social': #for social 
            df = df.drop(['time', 'utc_offset', 'responses', 'description','latx', 'longx', 'laty', 'longy', 'city', 'distance', 'weekday'], axis=1)
        elif dataType == 'semSpatSoc': #for semantic, spatial and social
            df = df.drop(['time', 'utc_offset', 'responses', 'city', 'distance', 'weekday'], axis=1)
        elif dataType == 'temporal':
            df = df.drop(['utc_offset', 'responses', 'group_id', 'description','latx', 'longx', 'laty', 'longy', 'city', 'distance'], axis=1)
        elif dataType == 'semSpatSocTemp':   
            df = df.drop(['utc_offset', 'responses', 'city', 'distance'], axis=1)
        

        #drop the old columns 
        df = df.drop(['event', 'user', 'response_nn', 'index'], axis=1)

        #analyze the data

        uniqueUsers, countsUsers = np.unique(df.user_id, return_counts=True)
        #add a column with the sum of user RSVPs
        df['userRSVP'] = countsUsers[df.user_id]

        uniqueEvents, countsEvents = np.unique(df.event_id, return_counts=True)
        
        #add a column with the sum of event RSVPs
        df['eventRSVP'] = countsEvents[df.event_id]
        df['rank'] = df.groupby(['event_id']).cumcount()+1

        #get 50% of the dataset as training data 
        #give us random values in [0,1] and if <0.5 return true , else false
        mask = round(df['eventRSVP']*0.6).astype('int64') >= df['rank']
        
        dfTraining = df[mask]
        dfTestingAndValidation = df[~mask]


        #true if user and event exists in training set, false otherwise
        mask = np.logical_and(np.isin(dfTestingAndValidation.user_id, dfTraining.user_id), np.isin(dfTestingAndValidation.event_id, dfTraining.event_id))
        dfTestingAndValidation = dfTestingAndValidation[mask]

        dfTraining = dfTraining.reset_index() 
        dfTestingAndValidation = dfTestingAndValidation.reset_index() 

        dfTraining = dfTraining.drop(['index', 'created', 'eventRSVP', 'userRSVP', 'rank'], axis=1)
        dfTestingAndValidation = dfTestingAndValidation.drop(['index', 'created', 'eventRSVP', 'userRSVP', 'rank'], axis=1)

       
        return dfTraining, dfTestingAndValidation

    def general_data(self, city):
        # #read the dataset
        # df = pd.read_csv('/home/nikoscha/Documents/ThesisR/datasets/dataset_new_new.csv', names=['user','response_nn', 'time', 'utc_offset','event','created','description', 'group_id','latx', 'longx','city' ,'laty', 'longy', 'distance', 'weekday'])
        # #response is 
        # # 0 for yes
        # # 1 for no
        # # 2 for waitlist

        # #delete unwanted row of columns
        # df = df.drop(df.index[0])

        # #drop cities we dont want
        # if city != 'all' :
        #     df = df[df.city == city]
        def converter(instr):
            return np.fromstring(instr[1:-1],sep=' ')
        
        #read the dataset
        if city == 'Chicago':
            df = pd.read_csv('/home/nikoscha/Documents/ThesisR/datasets/dataset_with_emb_Chicago.csv' ,converters={'description':converter}, names=['user','response_nn', 'time', 'utc_offset','event','created','description', 'group_id','latx', 'longx','city' ,'laty', 'longy', 'distance', 'weekday'])
        elif city == 'Phoenix':
            df = pd.read_csv('/home/nikoscha/Documents/ThesisR/datasets/dataset_with_emb_Phoenix.csv',converters={'description':converter} , names=['user','response_nn', 'time', 'utc_offset','event','created','description', 'group_id','latx', 'longx','city' ,'laty', 'longy', 'distance', 'weekday'])
        elif city == 'San Jose':
            df = pd.read_csv('/home/nikoscha/Documents/ThesisR/datasets/dataset_with_emb_San Jose.csv',converters={'description':converter} , names=['user','response_nn', 'time', 'utc_offset','event','created','description', 'group_id','latx', 'longx','city' ,'laty', 'longy', 'distance', 'weekday'])

        #delete unwanted row of columns
        df = df.drop(df.index[0])

        #response to int because response has string and int as dtypes
        df['responses'] = df['response_nn'].astype("int8")

        # waitlist is approximately 1.3 % so I will delete it 
        # df = df.loc[df.responses != 2]
        df = df[df.responses != 2]

        #delete no response, we just need 
        df = df[df.responses != 1]

        # set the yes response as number 1 instead of 0.
        df['responses'] = df.responses.replace(0, 1)

        #create numerical ids
        df['event_id'] = df['event'].astype("category").cat.codes
        df['user_id'] = df['user'].astype("category").cat.codes    
        df = df.sort_values(by=['event_id', 'created'])
        df = df.reset_index() 


            
        # Create a dictionary so we can get the original ids
        self.event_dictionary = df[['event_id', 'event']].drop_duplicates()
        self.event_dictionary['event_id'] = self.event_dictionary.event_id.astype(str)

        self.user_dictionary = df[['user_id','user']].drop_duplicates()
        self.user_dictionary['user_id'] = self.user_dictionary.user_id.astype(str)

        #drop the old columns 
        df = df.drop(['event', 'user', 'response_nn', 'index', 'utc_offset', 'city', 'distance'], axis=1)

        #analyze the data

        uniqueUsers, countsUsers = np.unique(df.user_id, return_counts=True)
        #add a column with the sum of user RSVPs
        df['userRSVP'] = countsUsers[df.user_id]

        uniqueEvents, countsEvents = np.unique(df.event_id, return_counts=True)
        
        #add a column with the sum of event RSVPs
        df['eventRSVP'] = countsEvents[df.event_id]
        df['rank'] = df.groupby(['event_id']).cumcount()+1

        #get 50% of the dataset as training data 
        #give us random values in [0,1] and if <0.5 return true , else false
        mask = round(df['eventRSVP']*0.6).astype('int64') >= df['rank']
        
        self.dfTraining = df[mask]
        self.dfTestingAndValidation = df[~mask]


        #true if user and event exists in training set, false otherwise
        mask = np.logical_and(np.isin(self.dfTestingAndValidation.user_id, self.dfTraining.user_id), np.isin(self.dfTestingAndValidation.event_id, self.dfTraining.event_id))
        self.dfTestingAndValidation = self.dfTestingAndValidation[mask]

        self.dfTraining = self.dfTraining.reset_index() 
        self.dfTestingAndValidation = self.dfTestingAndValidation.reset_index() 

        self.dfTraining = self.dfTraining.drop(['index', 'created', 'eventRSVP', 'userRSVP', 'rank'], axis=1)
        self.dfTestingAndValidation = self.dfTestingAndValidation.drop(['index', 'created', 'eventRSVP', 'userRSVP', 'rank'], axis=1)

        # Create lists of all events, users and respones
        self.events = list(np.sort(df.event_id.unique()))
        self.users = list(np.sort(df.user_id.unique()))      
        responses = list(self.dfTraining.responses)
        responsesTesting = list(self.dfTestingAndValidation.responses)
       
        # Get the rows and columns for our new matrix
        rows = self.dfTraining.user_id
        cols = self.dfTraining.event_id

        rowsTesting = self.dfTestingAndValidation.user_id
        colsTesting = self.dfTestingAndValidation.event_id

        # Contruct a sparse matrix for our users and events containing RSVPs 
        data_sparse = sp.csr_matrix((responses, (rows, cols)), shape=(len(self.users), len(self.events)))
        data_sparse_testing = sp.csr_matrix((responsesTesting, (rowsTesting, colsTesting)), shape=(len(self.users), len(self.events)))
        
        #get positive feedbacks
        # self.uids, self.eids = data_sparse.nonzero()
        self.puids, self.peids, values = sp.find(data_sparse == 1)

        # self.utids, self.etids = data_sparse_testing.nonzero()
        self.ptuids, self.pteids, valuesTesting = sp.find(data_sparse_testing == 1)


    def prepare_contextual_data(self, data):
        import tensorflow as tf

        import tensorflow_hub as hub
        
        hub_url = '/tmp/module/universal_module/'
        loaded = tf.saved_model.load(hub_url)

        embed = hub.KerasLayer(hub_url)

        #clean html tags etc from descriptions
        data['description'] = data.apply(lambda row: BeautifulSoup(row.description, features="html.parser").getText(), axis=1)

        #set description as 512 vector
        data['description'] = data.apply(lambda row: embed([row.description]), axis=1)
        return data
        
    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        # yield self.input[idx], self.y[idx]
