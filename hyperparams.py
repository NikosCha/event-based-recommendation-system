#HYPERPARAMS

epochs = 1000
batches = 50 #batches must be as the number of samples (SGD -> Batch size = 1 )
num_factors = 64 #latent features

#lambda regularizations 
lambda_user = 0.000001
lambda_event = 0.000001

learning_rate = 0.005

# The number of (u,i,j) triplets we sample for each batch.
samples = 1
