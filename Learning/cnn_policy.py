import tensorflow as tf
from stable_baselines.common.policies import FeedForwardPolicy

class CustomCnnPolicy(FeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        def network_fn(X, **kwargs):
            print(X.shape)

            if "network_kwargs" in kwargs.keys():
                params = kwargs["network_kwargs"]
            else:
                params = kwargs

            output_x = params["OUTPUT_X"]
            output_y = params["OUTPUT_Y"]
            num_hidden_layers = params["NUM_HIDDEN_LAYERS"]
            num_filters = params["NUM_FILTERS"]
            num_convs = params["NUM_CONVS"]
            filter_size = params["FILTER_SIZE"]
            stride = params["STRIDE"]


            conv_out = tf.layers.conv2d(
                inputs=X,
                filters=num_filters,
                kernel_size=[filter_size, filter_size],
                strides=stride,
                padding="valid",
                activation=tf.nn.relu,
                name="conv_initial"
            )

            for i in range(0, num_convs - 1):
                conv_out = tf.layers.conv2d(
                    inputs=conv_out,
                    filters=num_filters,
                    kernel_size=[filter_size, filter_size],
                    strides=stride,
                    padding="valid",
                    activation=tf.nn.relu,
                    name="conv_{}".format(i)
                )
            
            out = tf.layers.flatten(conv_out)
            for _ in range(num_hidden_layers):
                out = tf.layers.dense(out, output_x * output_y, activation=tf.nn.relu)

            print("Last layer conv network output shape", out.shape)

            return out

        super(CustomCnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse, cnn_extractor=network_fn, feature_extraction="cnn", **kwargs)

# make a cnn
# test grid size, filter size, step combinations
# we want to get the biggest grid we can