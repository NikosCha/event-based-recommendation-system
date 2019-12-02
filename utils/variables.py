import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import numpy as np

def init_variable(size, dim, name=None):
    #Initialize a new variable with random values 

    # return tf.Variable(tf.xavier_initializer(uniform=true))
    std = np.sqrt(2 / dim)
    return tf.Variable(tf.random.uniform([size, dim], -std, std), name=name)


def get_variable(graph, session, name):
    #get value of a TS variable
    v = graph.get_operation_by_name(name)
    v = v.values()[0]
    v = v.eval(session=session)

    return v