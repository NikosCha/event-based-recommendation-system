import numpy as np
import pandas as pd
import scipy.sparse as sp


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

    def events_rsvp_dataset(self):

        #read the dataset
        df = pd.read_csv('/home/nikoscha/Documents/ThesisR/dataset.csv', names=['response_nn','event','user','created'])
        #response is 
        # 0 for yes
        # 1 for no
        # 2 for waitlist

        #delete unwanted row of columns
        df = df.drop(df.index[0])

        #response to int because response has string and int as dtypes
        df['responses'] = df['response_nn'].astype("int8")

        # waitlist is approximately 1.3 % so I will delete it 
        # df = df.loc[df.responses != 2]
        df = df[df.responses != 2]

        # set the no response as number -1 instead of 1.
        df['responses'] = df.responses.replace(1, -1)

        # set the yes response as number 1 instead of 0.
        df['responses'] = df.responses.replace(0, 1)


        #delete negative responses
        # df = df[df.responses != -1]

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

        
    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        # yield self.input[idx], self.y[idx]
