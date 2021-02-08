import tensorflow as tf

def create_net(x, num_input, num_output, num_neurons, num_hidden_layers):
    weights = []
    for i in range(num_hidden_layers):
        with tf.compat.v1.variable_scope("Layer_" + str(i)):
            layer_in  = num_input if i == 0 else num_neurons
            layer_out = num_output if i == num_hidden_layers-1 else num_neurons
            add_relu  = i < num_hidden_layers-1
            x, w = _add_dense_layer(x, layer_in, layer_out, add_relu)
            weights.extend(w)
    return x, weights

def create_linear_regression(x, num_input, num_output):
    W = tf.compat.v1.get_variable("w", [num_input, num_output], initializer=tf.compat.v1.constant_initializer(1.0, dtype=tf.compat.v1.float64), dtype=tf.compat.v1.float64)
    return tf.compat.v1.matmul(x, W), W

def create_target_updates(weights, target_weights):
    init_updates = []
    for i in range(len(weights)):
        init_updates.append(tf.compat.v1.assign(target_weights[i], weights[i]))
    return init_updates

def _add_dense_layer(x, num_input, num_output, add_relu):
    W = tf.compat.v1.get_variable("w", [num_input, num_output], initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1, dtype=tf.compat.v1.float64), dtype=tf.compat.v1.float64)
    b = tf.compat.v1.get_variable("b", [num_output], initializer=tf.compat.v1.constant_initializer(0.1, dtype=tf.compat.v1.float64), dtype=tf.compat.v1.float64)
    x = tf.compat.v1.matmul(x, W) + b
    if add_relu:
        x = tf.compat.v1.nn.relu(x)
    return x, [W, b]
