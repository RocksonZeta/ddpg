import tensorflow as tf


def batch_norm(x,training , scope_name,activation = None):
	return tf.cond(training , 
		lambda :tf.contrib.layers.batch_norm(x,is_training=True, reuse=None, scale=True ,center=True,
		updates_collections =None,scope = scope_name , activation_fn = activation,decay=0.9, epsilon=1e-5),
		lambda :tf.contrib.layers.batch_norm(x,is_training=False,reuse=True, scale=True ,center=True,
		updates_collections =None,scope = scope_name, activation_fn = activation,decay=0.9, epsilon=1e-5)
	)