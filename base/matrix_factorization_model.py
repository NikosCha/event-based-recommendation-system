import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utils.variables import init_variable
from figures.diagrams import create_diagram


class MFModel:
    def __init__(self, graph, users, events, session ):
        self.graph = graph
        self.users = users
        self.events = events
        self.session = session
        self.cases = 2

    def build_model(self, num_factors, lambda_user, lambda_event, learning_rate):
        with self.graph.as_default():
            #loss function is -S ln s(xuij) + lw||W||^2
            #where s is sigmoid function
            #l is the regularization value
            #||W|| is the L2 norm


            #inputs in our model
            self.u = tf.compat.v1.placeholder(tf.int32, shape=(None, 1))
            self.i = tf.compat.v1.placeholder(tf.int32, shape=(None, 1))
            self.j = tf.compat.v1.placeholder(tf.int32, shape=(None, 1))


            u_f = init_variable(len(self.users), num_factors, name='user_factors')
            u_factors = tf.nn.embedding_lookup(u_f, self.u)

            event_factors = init_variable(len(self.events), num_factors, name='event_factors')
            i_factors = tf.nn.embedding_lookup(event_factors, self.i)
            j_factors = tf.nn.embedding_lookup(event_factors, self.j)

            # Calculate the dot product + bias for known and unknown
            # item to get xui and xuj.

            #check if matmul works
            xui = tf.reduce_sum(u_factors * i_factors, axis=2)
            xuj = tf.reduce_sum(u_factors * j_factors, axis=2)

            # We calculate xuij.
            xuij = xui - xuj

            # Calculate the mean AUC (area under curve).
            # if xuij is greater than 0, that means that 
            # xui is greater than xuj (and thats what we want).
            # u_auc = tf.reduce_mean(tf.to_float(xuij > 0))
            self.u_auc = tf.reduce_mean(tf.compat.v1.cast(xuij > 0, float))

            # Output the AUC value to tensorboard for monitoring.
            tf.compat.v1.summary.scalar('auc', self.u_auc)

            # Calculate the squared L2 norm ||W||**2 multiplied by l.

            #check if tf.norm works
            l2_norm = tf.add_n([
                lambda_user * tf.reduce_sum(tf.multiply(u_factors, u_factors)),
                lambda_event * tf.reduce_sum(tf.multiply(i_factors, i_factors)),
                lambda_event * tf.reduce_sum(tf.multiply(j_factors, j_factors)),
            ])



            # Calculate the loss as ||W||**2 - ln s(Xuij)
            #loss = l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(xuij)))
            self.loss = -tf.reduce_mean(tf.math.log(tf.sigmoid(xuij))) + l2_norm
            
            # Train using the Adam optimizer to minimize 
            # our loss function.
            opt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)
            self.step = opt.minimize(self.loss)

            # Initialize all tensorflow variables.
            init = tf.compat.v1.global_variables_initializer()
                
            self.session.run(init)

    def train_model(self, epochs, batches, puids, peids, nuids, neids, pvuids, pveids, nvuids, nveids, factors, samples): 
        progress = tqdm(total=batches*epochs)

        loss_array = []
        auc_array = []
        num_epochs = []
        for k in range(epochs):
            for _ in range(batches):

                # First we sample 15000 uniform indices.
                # idx = np.random.randint(low=0, high=len(puids), size=samples)

                # We then grab the users matching those indices.
                # batch_u = puids[idx].reshape(-1, 1)

                # Then the known items for those users.
                # batch_i = peids[idx].reshape(-1, 1)

                # Lastly we randomly sample one unknown item for each user.
                # batch_j = np.random.randint(
                        # low=0, high=len(self.events), size=(samples, 1), dtype='int32')


                #get a random int from 1 to 3 
                # i = np.random.randint(1, 3, dtype='int32')
                l = []
                auc = []
                for i in range(1,self.cases + 1):
                    batch_u, batch_i, batch_j = self.get_data(i, puids, peids, nuids, neids, samples//self.cases)

                    # Feed our users, known and unknown items to
                    # our tensorflow graph. 
                    feed_dict = { self.u: batch_u, self.i: batch_i, self.j: batch_j }
                    # We run the session.
                    _, _l, _auc = self.session.run([self.step, self.loss, self.u_auc], feed_dict)
                    l.append(_l)
                    auc.append(_auc)

                


            progress.update(batches)
            progress.set_description('Loss: %.3f | AUC: %.3f' % (np.mean(l), np.mean(auc)))
            #validate our model
            loss_epoch, auc_epoch = self.validate_model(pvuids, pveids, nvuids, nveids, samples)
            loss_array.append(loss_epoch)
            auc_array.append(auc_epoch)
            num_epochs.append(k)
        progress.close()
        create_diagram(num_epochs, loss_array, auc_array, 'Number of Epochs', 'Loss', 'AUC', '', '', 'Loss_AUC_validation_' + str(factors) + 'factors.png')

    
    def validate_model(self, putids, petids, nutids, netids, samples):
        # tidx = np.random.randint(low=0, high=len(utids), size=samples)

        # # We then grab the users matching those indices.
        # test_u = utids[tidx].reshape(-1, 1)

        # # Then the known items for those users.
        # test_i = etids[tidx].reshape(-1, 1)

        # # Lastly we randomly sample one unknown item for each user.
        # test_j = np.random.randint(
        #         low=0, high=len(self.events), size=(samples, 1), dtype='int32')

        l = []
        auc = []
        for i in range(1,self.cases + 1):
            batch_u, batch_i, batch_j = self.get_data(i, putids, petids, nutids, netids, samples//self.cases)

            # Feed our users, known and unknown items to
            # our tensorflow graph. 
            feed_dict = { self.u: batch_u, self.i: batch_i, self.j: batch_j }
            # We run the session.
            _l, _auc = self.session.run([self.loss, self.u_auc], feed_dict)
            l.append(_l)
            auc.append(_auc)

        # # Feed our users, known and unknown items to
        # # our tensorflow graph. 
        # feed_dict = { self.u: test_u, self.i: test_i, self.j: test_j }

        # # We run the session.
        # l, auc = self.session.run([self.loss, self.u_auc], feed_dict) 

        return np.mean(l), np.mean(auc)

    def get_data(self, i, puids, peids, nuids, neids, samples):
        #we have 3 cases:
        #i = 1 -> known positives (yes rsvp -> 1 value) vs unknown(0 value) (user prefers positives)
        #i = 2 -> known positives (yes rsvp -> 1 value) vs known negatives(no rsvp -> -1 value) (user prefers positives)
        #i = 3 -> unknown(0 value) vs known negatives(no rsvp -> -1 value) (user prefers unknown)
        
        if i == 1 :
            # First we sample 15000 uniform indices.
            idx = np.random.randint(low=0, high=len(puids), size=samples)

            # We then grab the users matching those indices.
            batch_u = puids[idx].reshape(-1, 1)

            # Then the known items for those users.
            batch_i = peids[idx].reshape(-1, 1)

            # Lastly we randomly sample one unknown item for each user.
            batch_j = np.random.randint(
                    low=0, high=len(self.events), size=(samples, 1), dtype='int32')


        elif i == 2 :
            # First we sample 15000 uniform indices.
            pidx = np.random.randint(low=0, high=len(puids), size=samples)

            # We then grab the users matching those indices.
            batch_u = puids[pidx].reshape(-1, 1)

            #we need to take events for the users we choose, so we find the users who has negative rsvps 

            #values which exists in both arrays
            users_with_negatives = np.intersect1d(puids[pidx], nuids)
            #mask -> true when value exists 
            mask = np.in1d(nuids, users_with_negatives)
            
            #get as many events as the sample
            negative_user_events = neids[mask]

            if len(negative_user_events) >= samples :
                id = np.random.randint(low=0, high=len(negative_user_events), size=samples)
                negative_user_events = negative_user_events[id]
            elif len(negative_user_events) < samples :
                #if we dont have as many negatives as the sample , we fill it in with unknown.
                unknown_events = np.random.randint(
                    low=0, high=len(self.events), size=(samples-len(negative_user_events)), dtype='int32')
                negative_user_events = np.concatenate((negative_user_events,unknown_events),axis=None)
     
            # Then the known items for those users.
            batch_i = peids[pidx].reshape(-1, 1)

            batch_j = negative_user_events.reshape(-1, 1)


        elif i == 3 :
            # First we sample 15000 uniform indices.
            idx = np.random.randint(low=0, high=len(nuids), size=samples)

            # We then grab the users matching those indices.
            batch_u = nuids[idx].reshape(-1, 1)

            # we randomly sample one unknown item for each user.
            batch_i = np.random.randint(
                    low=0, high=len(self.events), size=(samples, 1), dtype='int32')

            # Then the known items for those users.
            batch_j = neids[idx].reshape(-1, 1)

        return batch_u, batch_i, batch_j

