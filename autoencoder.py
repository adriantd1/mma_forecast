from fmcrawler_sql import *
from utils import *

epochs = 100000
batch_size = 256
eval_step = 10

# data = get_rounds()

data_list = format_data(get_fighters())

data = []

for input in data_list:
    for fight in input['fight']:
        data.append(np.asarray(fight))
        arr = np.asarray(fight)

data = np.asarray(data)

mins = np.min(data, axis=0)
maxs = np.max(data, axis=0)
rng = maxs - mins

data = 1.0 - ((1.0 * (maxs - data)) / rng)

data = np.expand_dims(data, axis=4)

train_data = data[:int(len(data)*0.8)]
valid_data = data[int(len(data)*0.8):]

train_dataset = tf.data.Dataset.from_tensor_slices(train_data).batch(batch_size)
valid_dataset = tf.data.Dataset.from_tensor_slices(valid_data).batch(batch_size)

train_dataset.repeat()
valid_dataset.repeat()

train_iterator = train_dataset.make_initializable_iterator()
train_next_element = train_iterator.get_next()

valid_iterator = valid_dataset.make_initializable_iterator()
valid_next_element = valid_iterator.get_next()

learning_rate = 0.0001
inputs = tf.placeholder(tf.float32, (None, 5, 2, 22, 1), name='inputs')
targets = tf.placeholder(tf.float32, (None, 5, 2, 22, 1), name='targets')


def new_weights(shape, name='weights'):
    initializer = tf.contrib.layers.xavier_initializer()
    w = tf.Variable(initializer(shape), name=name)
    return w


def new_biases(length, name='bias'):
    b = tf.Variable(tf.constant(0.05, shape=[length]), name=name)
    return b


def encode(input):
    ### Encoder
    conv1 = tf.layers.conv3d(inputs=input, filters=32, kernel_size=(2, 2, 2), padding='same', activation=tf.nn.relu)
    # Now 5x20x32

    conv2 = tf.layers.conv3d(inputs=conv1, filters=64, kernel_size=(3, 2, 5), strides=(2, 2, 2), padding='same', activation=tf.nn.relu)
    # Now 3x9x32

    print('Conv2 shape')
    print(conv2.shape)

    # conv3 = tf.layers.conv2d(inputs=conv2, filters=16, kernel_size=(3,3), strides=(1,3), padding='valid', activation=tf.nn.relu)
    # Now 1x3x16

    weights_3 = new_weights(shape=[2, 1, 9, 64, 8])

    biases = new_biases(length=8)

    conv3 = tf.nn.conv3d(conv2, filter=weights_3, strides=[1, 2, 1, 3, 1], padding='SAME')

    layer = conv3 + biases

    layer = tf.nn.relu(layer, name='encoded')

    print('Layer shape')
    print(layer.shape)

    return layer


def decode(encoded):
    ### Decoder
    # upsample1 = tf.image.resize_images(encoded, size=(3,9), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    weights = new_weights(shape=[2, 1, 9, 64, 8])

    upsample = tf.nn.conv3d_transpose(value=encoded, filter=weights,
                                      output_shape=(batch_size, 3, 1, 11, 64), strides=[1, 2, 1, 3, 1],
                                      padding='SAME', name='reconstructed_image')

    print('Upsample shape')
    print(upsample.shape)
    # Now 3x9x16
    # conv4 = tf.layers.conv2d(inputs=upsample, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 3x9x16
    # upsample2 = tf.image.resize_images(conv4, size=(5,20), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    weights2 = new_weights(shape=[3, 2, 5, 32, 64])
    upsample2 = tf.nn.conv3d_transpose(value=upsample, filter=weights2,
                                      output_shape=(batch_size, 5, 2, 22, 32), strides=[1, 2, 2, 2, 1],
                                      padding='SAME', name='upsample2')

    print('Upsample2 shape')
    print(upsample2.shape)

    conv5 = tf.layers.conv3d(inputs=upsample2, filters=1, kernel_size=(2, 2, 2), padding='same', activation=tf.nn.relu)

    # weights_6 = new_weights(shape=[5,20,32, 1])
    #
    # biases = new_biases(length=1)
    #
    # conv6 = tf.nn.conv2d(conv5, filter=weights_6, strides=[1, 1, 1, 1], padding='SAME')
    #
    # layer = conv6 + biases
    #
    # layer = tf.nn.relu(layer, name='last_layer')

    return conv5

encoder = encode(inputs)

decoded = decode(encoder)

print('Decoded shape')
print(decoded.shape)

# Pass logits through sigmoid to get reconstructed image
sig_decoded = tf.nn.sigmoid(decoded, name='decoded')

# Pass logits through sigmoid and calculate the cross-entropy loss
loss = tf.losses.mean_squared_error(labels=targets, predictions=decoded)

max_loss = tf.reduce_max(loss)

# Get cost and define the optimizer
cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Saver
saver = tf.train.Saver()


def train():
    sess = tf.Session()
    # Set's how much noise we're adding to the MNIST images
    noise_factor = 0.5
    sess.run(tf.global_variables_initializer())

    best_avg_cost = 1000

    counter = 0

    for e in range(epochs):
        sess.run(train_iterator.initializer)
        avg_cost = 0
        for ii in range(int(len(train_data) // batch_size)):
            batch = sess.run(train_next_element)

            # # Get images from the batch
            # imgs = batch[0].reshape((-1, 28, 28, 1))
            #
            # # Add random noise to the input images
            # noisy_imgs = imgs + noise_factor * np.random.randn(*imgs.shape)
            # # Clip the images to be between 0 and 1
            # noisy_imgs = np.clip(noisy_imgs, 0., 1.)

            # Noisy images as inputs, original images as targets
            batch_cost, _ = sess.run([cost, opt], feed_dict={inputs: batch,
                                                             targets: batch})
            avg_cost += batch_cost

        avg_cost /= int(len(train_data) // batch_size)
        print("Epoch: {}/{}...".format(e + 1, epochs),
              "Training loss: {:.6f}".format(avg_cost))

        if e % eval_step == 0:
            sess.run(valid_iterator.initializer)
            val_avg_cost = 0
            for ii in range(int(len(valid_data) // batch_size)):
                valid_batch = sess.run(valid_next_element)

                val_batch_cost, m_loss = sess.run([cost, max_loss], feed_dict={inputs: valid_batch,
                                                                 targets: valid_batch})
                val_avg_cost += val_batch_cost

            val_avg_cost /= int(len(valid_data) // batch_size)


            print()
            print('##########################\n')
            print("Epoch: {}/{}...".format(e + 1, epochs),
                  "Validation loss: {:.6f}".format(val_avg_cost))
            print('Max loss: %.6f' % m_loss)

            if val_avg_cost <= best_avg_cost:
                save_path = saver.save(sess, "saves/model")
                best_avg_cost = val_avg_cost
                counter = 0
            else:
                counter += 1
                print('Incrementing counter to %d' % counter)
                print('Best avg loss: %.4f' % best_avg_cost)
                print('Max loss: %.6f' % m_loss)
            print()
            print('##########################\n')

        if counter >= 10:
            print('Training end, best loss is %.6f' % best_avg_cost)
            break

if __name__ == '__main__':
    train()
