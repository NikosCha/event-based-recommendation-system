def get_map_from_files(): 
    import os
    import pandas as pd
    import json
    
    folderNames = []
    folderAverages = []
    for dirpath, dirnames, files in os.walk('/home/nikoscha/Documents/ThesisR/datasets/Rankings/'):
        print(f'Found directory: {dirpath}')

        usersAveragePrecision = []
        if len(files) > 0:
            for file_name in files:
                df = pd.read_csv(dirpath + '/' + file_name, names=['Score','TruePositive'])
                if df.empty: continue
                df = df.drop(df.index[0])
                df = df.reset_index()
                precision = get_precision(df)
                usersAveragePrecision.append(get_average(precision))
            folderAverages.append(get_average(usersAveragePrecision))
            folderNames.append(dirpath.split('/')[-1])

    dfData = {'Folder': folderNames, 'Precision': folderAverages}
    exportDF = pd.DataFrame(dfData) 
    exportDF.to_csv('/home/nikoscha/Documents/ThesisR/datasets/Rankings/Precisions.csv', index=False)


def get_precision(df):
    truePositives = 0
    total = 0 
    precision = []
    for index, row in df.iterrows():
        total += 1  
        if int(float(row['TruePositive'])) == 1 :
            truePositives += 1
            precision.append(round(truePositives / total, 5))
    return precision

def get_average(array):
    import numpy as np
    average = np.mean(array)
    return average 

def get_NDCG_from_files(): 
    import os
    import pandas as pd
    import json
    
    folderNames = []
    numOfUsers = []
    folderAverages_5 = []
    folderAverages_10 = []

    for dirpath, dirnames, files in os.walk('/home/nikoscha/Documents/ThesisR/datasets/Rankings/San_Jose_less_cat'):
        print(f'Found directory: {dirpath}')

        NDCG_5 = []
        NDCG_10 = []
        if len(files) > 0:
            for file_name in files:
                df = pd.read_csv(dirpath + '/' + file_name, names=['Score','TruePositive'])
                if df.empty: continue
                df = df.drop(df.index[0])
                df = df.reset_index()

                NDCG_5.append(get_DCG(5, df)/get_IDCG(5, df)) 
                NDCG_10.append(get_DCG(10, df)/get_IDCG(10, df)) 

            folderAverages_5.append(get_average(NDCG_5))
            folderAverages_10.append(get_average(NDCG_10))
            numOfUsers.append(len(files))
            folderNames.append(dirpath.split('/')[-1])

    dfData = {'Folder': folderNames, 'NDCG': folderAverages_5, 'NumOfUsers': numOfUsers}
    exportDF = pd.DataFrame(dfData) 
    exportDF.to_csv('/home/nikoscha/Documents/ThesisR/datasets/Rankings/NDCG_5.csv', index=False)

    dfData = {'Folder': folderNames, 'NDCG': folderAverages_10, 'NumOfUsers': numOfUsers}
    exportDF = pd.DataFrame(dfData) 
    exportDF.to_csv('/home/nikoscha/Documents/ThesisR/datasets/Rankings/NDCG_10.csv', index=False)

def get_DCG(n, df):
    import numpy as np
    DCG=0
    for index, row in df.iterrows():
        if int(float(row['TruePositive'])) == 1 :
            #DCG = rel(i)/log(index + 1) but index in here starts from 0 so index + 2 
            DCG = DCG + 1/np.log(index + 2)
        if (index + 1) == n: break 
    return DCG

def get_IDCG(n, df):    
    import numpy as np
    IDCG=0
    df['TruePositive'] = df['TruePositive'].astype('float').astype('int32')
    truePositives = df[df['TruePositive'] == 1].shape[0]
    for index in range(1,truePositives+1):
        IDCG += 1/np.log(index + 1)
        if (index) == n: break 
    return IDCG

# get_map_from_files()
get_NDCG_from_files()