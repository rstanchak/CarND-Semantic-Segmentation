import os.path
import tensorflow as tf
import numpy as np
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out

#tests.test_load_vgg(load_vgg, tf)

def conv_1x1(x, num_outputs, name=None):
    return tf.layers.conv2d(x, num_outputs, kernel_size=1, kernel_initializer=tf.truncated_normal_initializer(), name=name)

def get_bilinear_filter(filter_shape, upscale_factor):
    """
    http://cv-tricks.com/image-segmentation/transpose-convolution-in-tensorflow/
    """
    ##filter_shape is [width, height, num_in_channels, num_out_channels]
    kernel_size = filter_shape[1]
    ### Centre location of the filter for which value is calculated
    if kernel_size % 2 == 1:
        centre_location = upscale_factor - 1
    else:
        centre_location = upscale_factor - 0.5

    bilinear = np.zeros([filter_shape[0], filter_shape[1]])
    for x in range(filter_shape[0]):
        for y in range(filter_shape[1]):
            ##Interpolation Calculation
            value = (1 - abs((x - centre_location)/ upscale_factor)) * (1 - abs((y - centre_location)/ upscale_factor))
            bilinear[x, y] = value
    weights = np.zeros(filter_shape)
    for i in range(filter_shape[2]):
        weights[:, :, i, i] = bilinear
    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)

    bilinear_weights = tf.get_variable(name="decon_bilinear_filter", initializer=init,
                           shape=weights.shape)
    return bilinear_weights

def upsample_layer(bottom, n_channels, name, upscale_factor):
    """
    http://cv-tricks.com/image-segmentation/transpose-convolution-in-tensorflow/
    """

    kernel_size = 2*upscale_factor - upscale_factor%2
    stride = upscale_factor
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        # Shape of the bottom tensor
        in_shape = tf.shape(bottom)

        h = in_shape[1] * stride
        w = in_shape[2] * stride
        new_shape = [in_shape[0], h, w, n_channels]
        output_shape = tf.stack(new_shape)

        filter_shape = [kernel_size, kernel_size, n_channels, n_channels]

        weights = get_bilinear_filter(filter_shape,upscale_factor)
        deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                        strides=strides, padding='SAME')

    return deconv

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    skip_layer3 = conv_1x1(vgg_layer3_out, num_classes, "skip-layer3")
    skip_layer4 = conv_1x1(vgg_layer4_out, num_classes, "skip-layer4")
    bottom = conv_1x1(vgg_layer7_out, num_classes, "latent-space")

    up1 = upsample_layer( bottom, num_classes, "upsample1", 2 ) + skip_layer4 
    up2 = upsample_layer( up1, num_classes, "upsample2", 2 ) + skip_layer3

    layers_out = [up2]
    for i in range(2,5):
        up = upsample_layer( layers_out[-1], num_classes, "upsample-%d" % i, 2 )
        layers_out.append( up )
    return layers_out[-1]

#tests.test_layers(layers)

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, [-1, num_classes])
    labels = tf.reshape(correct_label, [-1, num_classes])

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy')
    loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    """
    with tf.name_scope('metrics'):
        label = tf.cast(tf.argmax(labels, 1), tf.float32)
        prediction = tf.cast(tf.argmax(logits, 1), tf.float32)
        tf.summary.image('label', tf.reshape(label, [shape[0], shape[1], shape[2], 1]))
        tf.summary.image('prediction', tf.reshape(prediction, [shape[0], shape[1], shape[2], 1]))

        correct_prediction = tf.equal(label, prediction)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        precision, _= tf.metrics.precision(label, prediction)
        recall, _ = tf.metrics.recall(label, prediction)
        tf.summary.scalar('precision', precision)
        tf.summary.scalar('recall', recall)
    """

    return logits, optimizer, loss

#tests.test_optimize(optimize)
def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, metrics=None, writer=None):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())


    with tf.name_scope('metrics'):
        tf.summary.scalar("cost", cross_entropy_loss)
        tf.summary.scalar("accuracy", metrics.accuracy)
        tf.summary.scalar("specificity", metrics.specificity)
        tf.summary.scalar("recall", metrics.recall)

    summary_op = tf.summary.merge_all()

    idx = 0
    for i in range(epochs):
        epoch_loss, epoch_accuracy, epoch_specificity, epoch_recall, epoch_N = 0,0,0,0,0
        for batch_x, batch_y in get_batches_fn(batch_size):
            feed_dict = {input_image:batch_x, keep_prob:0.5, correct_label:batch_y}
            _, loss, summary, accuracy, specificity, recall = sess.run([train_op, cross_entropy_loss, summary_op, metrics.accuracy, metrics.specificity, metrics.recall], feed_dict=feed_dict)
            N = batch_x.shape[0]
            epoch_loss += loss
            epoch_accuracy += accuracy*N
            epoch_specificity += specificity*N
            epoch_recall += recall*N
            epoch_N += N
            if idx % 10 == 0:
                print('Batch: {:4d}\tLoss: {:0.3f}\tAccuracy: {:0.3f}\tSpecificity: {:0.3f}\tRecall: {:0.3f}'.format(idx, loss, accuracy, specificity, recall))
            if writer != None:
                writer.add_summary(summary, idx)
                idx += 1
        epoch_loss /= epoch_N
        epoch_accuracy /= epoch_N
        epoch_specificity /= epoch_N
        epoch_recall /= epoch_N
        print('Epoch {:4d}\tLoss: {:0.3f}\tAccuracy: {:0.3f}\tSpecificity: {:0.3f}\tRecall: {:0.3f}'.format(i, epoch_loss, epoch_accuracy, epoch_specificity, epoch_recall))
    pass
#tests.test_train_nn(train_nn)

class MyMetrics(object):
    def __init__( self, correct_label, logits, num_classes):
        """
        http://ronny.rest/blog/post_2017_09_11_tf_metrics/
        """
        labels = tf.reshape(correct_label, [-1, num_classes])
        label = tf.cast(tf.argmax(labels, 1), tf.float32)
        prediction = tf.cast(tf.argmax(logits, 1), tf.float32)

        with tf.name_scope("my_metrics"):
            confusion = tf.cast(tf.confusion_matrix( label, prediction ), tf.float32)
            TP = confusion[1, 1]
            TN = confusion[0, 0]
            FP = confusion[0, 1]
            FN = confusion[1, 0]
            self.recall = TP/(TP+FP)
            self.specificity = TN/(TN+FN)
            self.accuracy = (TP+TN)/(TP+TN+FN+FP)


def run():
    epochs = 10
    batch_size = 4
    learning_rate = 0.001
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    logs_path = './logdir'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        correct_label  = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes], name='correct_label')

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        my_metrics = MyMetrics(correct_label, logits, num_classes)

        writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input, correct_label, keep_prob, learning_rate, my_metrics, writer)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
