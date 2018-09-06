import json
import matplotlib
import shutil

from fmcrawler_sql import *
from utils import *
from config import *

matplotlib.use('Agg')

from matplotlib import pyplot as plt
import time

sess = tf.Session()

input_sheet, fighters_stat, labels, methods, centroids = add_centroids(get_data_with_labels(transform_data(format_data(get_fighters()), sess)), sess)

# Keep data that is in the most populated centroid
# filter_perm = [i for i, j in enumerate(centroids) if j == cc]
#
# input_sheet = input_sheet[filter_perm]
# fighters_stat = fighters_stat[filter_perm]
# labels = labels[filter_perm]
# methods = methods[filter_perm]
# centroids = centroids[filter_perm]

# Randomize the data
permutations = np.random.permutation(input_sheet.shape[0])

input_sheet = input_sheet[permutations]
fighters_stat = fighters_stat[permutations]
labels = labels[permutations]
methods = methods[permutations]
centroids = centroids[permutations]

split1 = int(len(permutations) * 0.8)
split2 = int(len(permutations) * 0.9)

train_sheet, train_stats, train_labels, train_methods, train_centroids = augment_data(input_sheet[:split1],
                                                                                      fighters_stat[:split1],
                                                                                      labels[:split1],
                                                                                      methods[:split1],
                                                                                      centroids[:split1],
                                                                                      10)
valid_sheet, valid_stats, valid_labels, valid_methods, valid_centroids = input_sheet[split1:split2],\
                                                                         fighters_stat[split1:split2],\
                                                                         labels[split1:split2],\
                                                                         methods[split1:split2],\
                                                                         centroids[split1:split2]
test_sheet, test_stats, test_labels, test_methods, test_centroids = input_sheet[split2:], \
                                                                    fighters_stat[split2:], \
                                                                    labels[split2:], \
                                                                    methods[split2:], \
                                                                    centroids[split2:]

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_sheet, train_stats, train_labels, train_methods, train_centroids)).batch(batch_size).shuffle(buffer_size=2000)
valid_dataset = tf.data.Dataset.from_tensor_slices(
    (valid_sheet, valid_stats, valid_labels, valid_methods, valid_centroids)
).batch(len(valid_stats)).shuffle(buffer_size=2000)
test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_sheet, test_stats, test_labels, test_methods, test_centroids)).batch(len(test_stats)).shuffle(buffer_size=2000)

train_dataset.repeat()
valid_dataset.repeat()
test_dataset.repeat()

# create general iterator
iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes, )
next_element = iterator.get_next()

training_init_op = iterator.make_initializer(train_dataset)
validation_init_op = iterator.make_initializer(valid_dataset)
testing_init_op = iterator.make_initializer(test_dataset)

# Placeholders
inputs = tf.placeholder(tf.float32, (None, 5, 48, 1), name='in')
stats = tf.placeholder(tf.float32, (None, 16), name='stats')
targets = tf.placeholder(tf.float32, (None, 2), name='targ')
target_method = tf.placeholder(tf.float32, (None, 9), name='target_method')
is_training = tf.placeholder(tf.bool, name='training')

keep_prob = tf.placeholder(tf.float32, name='keep_prob')

lr = tf.placeholder(tf.float32, name='lr')

print(len(input_sheet))


def model(inputs, stats, keep_prob, training):
    with tf.name_scope('model') as scope:
        pool0 = tf.nn.dropout(inputs, 0.8)

        conv0 = tf.layers.conv2d(inputs=pool0, filters=128, kernel_size=(3, 1), strides=(1, 1), padding='same',
                                 activation=tf.nn.elu, name=scope + 'conv0')

        print(conv0.shape)
        # print(tf.get_variable('model/conv0/kernel:0').shape)

        bn0 = tf.contrib.layers.batch_norm(conv0, center=True, scale=True, is_training=training)

        pool05 = tf.nn.dropout(bn0, keep_prob=keep_prob)

        w = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)[0]
        b = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)[1]
        tf.summary.histogram('weights', w)
        tf.summary.histogram('bias', b)

    with tf.name_scope('model') as scope:
        # Input is 5x48x1
        conv1 = tf.layers.conv2d(inputs=pool05, filters=256, kernel_size=(1, 8), strides=(1, 1), padding='same',
                                 activation=tf.nn.elu, name=scope + 'conv1')
        print(conv1.shape)
        # Now dim is 5x22x64


        bn1 = tf.contrib.layers.batch_norm(conv1, center=True, scale=True, is_training=training)

        pool1 = tf.nn.dropout(bn1, keep_prob=keep_prob)

        w = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)[0]
        b = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)[1]
        tf.summary.histogram('weights', w)
        tf.summary.histogram('bias', b)

    with tf.name_scope('model') as scope:
        conv2 = tf.layers.conv2d(inputs=pool1, filters=512, kernel_size=(3, 4), strides=(1, 2), padding='valid',
                                 activation=tf.nn.elu, name=scope + 'conv2')
        print(conv2.shape)
        # Now dim is 3x10x128

        bn2 = tf.contrib.layers.batch_norm(conv2, center=True, scale=True, is_training=training)

        pool2 = tf.nn.dropout(bn2, keep_prob=keep_prob)

        w = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)[0]
        b = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)[1]
        tf.summary.histogram('weights', w)
        tf.summary.histogram('bias', b)

    with tf.name_scope('model') as scope:
        conv3 = tf.layers.conv2d(inputs=pool2, filters=1024, kernel_size=(2, 3), strides=(1, 2), padding='valid',
                                 activation=tf.nn.elu, name=scope + 'conv3')
        print(conv3.shape)

        pool3 = tf.nn.dropout(conv3, keep_prob=keep_prob)

        w = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)[0]
        b = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)[1]
        tf.summary.histogram('weights', w)
        tf.summary.histogram('bias', b)


    flat = tf.layers.flatten(pool3)

    bn3 = tf.contrib.layers.batch_norm(flat, center=True, scale=True, is_training=training)

    layer_fc0 = new_fc_layer(bn3,
                             num_inputs=flat.shape[1].value,
                             num_outputs=512,
                             keep_prob=keep_prob,
                             use_elu=True,
                             use_dropout=True,
                             layer_name='fc0'
                             )

    layer_with_stats = tf.concat((layer_fc0, stats), axis=1)

    bn4 = tf.contrib.layers.batch_norm(layer_with_stats, center=True, scale=True, is_training=training)

    layer_fc1 = new_fc_layer(bn4,
                             num_inputs=layer_with_stats.shape[1].value,
                             num_outputs=256,
                             keep_prob=keep_prob,
                             use_elu=True,
                             use_dropout=True,
                             layer_name='fc1'
                             )

    layer_fc2 = new_fc_layer(layer_fc1,
                             num_inputs=layer_fc1.shape[1].value,
                             num_outputs=128,
                             keep_prob=keep_prob,
                             use_elu=True,
                             use_dropout=True,
                             layer_name='fc2'
                             )

    # bn5 = tf.layers.batch_normalization(layer_fc2, center=True, scale=True, training=True)

    layer_fc3 = new_fc_layer(layer_fc2,
                             num_inputs=128,
                             num_outputs=64,
                             keep_prob=keep_prob,
                             use_elu=True,
                             use_dropout=False,
                             layer_name='fc3'
                             )

    layer_pred = new_fc_layer(input=layer_fc3,
                              num_inputs=64,
                              num_outputs=2,
                              use_elu=False,
                              use_dropout=False,
                              layer_name='prediction')

    layer_method = new_fc_layer(input=layer_fc3,
                                num_inputs=64,
                                num_outputs=9,
                                use_elu=False,
                                use_dropout=False,
                                layer_name='method_pred')

    return layer_pred, layer_method


prediction, method = model(inputs, stats, keep_prob, is_training)

y_true_cls = tf.argmax(targets, axis=1, name='y_true_cls')

method_true_cls = tf.argmax(methods, axis=1, name='method_true_cls')

# Cost function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=targets)
# method_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=method, labels=target_method)
# cost = tf.reduce_mean(tf.add(cross_entropy, tf.reshape(tf.convert_to_tensor([method_cross_entropy, method_cross_entropy]), (-1, 2))), name='cost')
cost = tf.reduce_mean(cross_entropy)
# cost = tf.reduce_mean(tf.add(cross_entropy, method_cross_entropy), name='cost')

y_pred = tf.nn.softmax(prediction, name='y_pred')

y_pred_cls = tf.argmax(y_pred, axis=1, name='y_pred_cls')

# Performance measure
correct_prediction = tf.equal(y_pred_cls, y_true_cls, name='Correct_pred')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='acc')

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(lr)

    grads_and_vars = optimizer.compute_gradients(cost)

    opt = optimizer.minimize(cost)

# Tf placeholders
cost_ph = tf.placeholder(tf.float32)
acc_ph = tf.placeholder(tf.float32)

tf.summary.scalar('Loss', cost_ph)
tf.summary.scalar('Accuracy', acc_ph)

for g,v in grads_and_vars:
    if 'fc3' in v.name and 'kernel' in v.name:
        with tf.name_scope('gradients'):
            tf_last_grad_norm = tf.sqrt(tf.reduce_mean(g**2))
            tf.summary.histogram('grad_norm', tf_last_grad_norm)
            break

# Saver
saver = tf.train.Saver()

def simulate_fights(fight_list):
    sess = tf.Session()

    saver = tf.train.import_meta_graph('pred_saves/model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('pred_saves/'))

    graph = tf.get_default_graph()
    result = graph.get_tensor_by_name('y_pred_cls:0')

    t_inputs, t_stats = graph.get_tensor_by_name('in:0'), graph.get_tensor_by_name('stats:0')

    results = []

    nb_iterations = 100


    for fighters in fight_list:
        f1, f2 = fighters
        data = simulated_fight_data(f1, f2)
        data = format_data(data, simulating=True)
        data = transform_data(data)
        stats = get_fighter_stats()

        v = data[f1]
        fighter_sheet = []
        for f in v[-5:len(v)]:
            fighter_sheet.append(f[0])
        f_stat = stats.get(f1, None)
        opp_stat = stats.get(f2, None)
        if f_stat is None or opp_stat is None:
            print('Skiping fighter %s' % f1)
            return
        stat = np.subtract(f_stat, opp_stat)

        feed_dict = {t_inputs: np.expand_dims(np.expand_dims(np.asarray(fighter_sheet),0),3), t_stats: np.expand_dims(stat, 0)}

        wins = 0
        for i in range(nb_iterations):
            winner = sess.run(result, feed_dict=feed_dict)
            wins += winner[0]

        results.append('Prediction for %s: %s out of %d' % (f1, wins, nb_iterations))

        f2, f1 = fighters
        data = simulated_fight_data(f1, f2)
        data = format_data(data, simulating=True)
        data = transform_data(data)
        stats = get_fighter_stats()

        v = data[f1]
        fighter_sheet = []
        for f in v[-5:len(v)]:
            fighter_sheet.append(f[0])
        f_stat = stats.get(f1, None)
        opp_stat = stats.get(f2, None)
        if f_stat is None or opp_stat is None:
            print('Skiping fighter %s' % f1)
            return
        stat = np.subtract(f_stat, opp_stat)

        feed_dict = {t_inputs: np.expand_dims(np.expand_dims(np.asarray(fighter_sheet), 0), 3),
                     t_stats: np.expand_dims(stat, 0)}

        wins = 0
        for i in range(nb_iterations):
            winner = sess.run(result, feed_dict=feed_dict)
            wins += winner[0]

        results.append('Prediction for %s: %s out of %d' % (f1, wins, nb_iterations))

    for r in results:
        print(r)

    sess.close()


def train(threshold, l_rate, it, exp_dir):
    sess = tf.Session()

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(exp_dir + '/train_' + str(it),
                                         sess.graph)
    test_writer = tf.summary.FileWriter(exp_dir + '/test_' + str(it))

    # Set's how much noise we're adding to the MNIST images
    sess.run(tf.global_variables_initializer())

    best_avg_acc = 0

    counter = 0

    train_loss_curve, val_loss_curve, train_acc_curve, val_acc_curve = [], [], [], []

    time.sleep(3)

    for e in range(epochs):
        sess.run(training_init_op)
        avg_cost = 0
        avg_acc = 0
        # for ii in range(int(len(input_sheet[:split1]) // batch_size)):
        for ii in range(len(train_sheet) // batch_size):

            fights, stat, label, m, _ = sess.run(next_element)

            # Noisy images as inputs, original images as targets
            batch_cost, acc, train_preds,  _ = sess.run([cost, accuracy, y_pred, opt], feed_dict={inputs: np.expand_dims(fights, 3),
                                                                            stats: stat,
                                                                            targets: label,
                                                                            target_method: m,
                                                                            lr: l_rate,
                                                                            keep_prob: prob,
                                                                            is_training: 1})
            avg_cost += batch_cost
            avg_acc += acc
            #
            # if ii < 1:
            #     pprint(train_preds[:20])

        # avg_cost /= (len(input_sheet[:split1]) // batch_size)
        # avg_acc /= (len(input_sheet[:split1]) // batch_size)

        avg_cost /= (len(train_sheet) // batch_size)
        avg_acc /= (len(train_sheet) // batch_size)

        train_loss_curve.append(avg_cost)
        train_acc_curve.append(avg_acc)

        if e % 5 == 0:
            print("Epoch: {}/{}...".format(e + 1, epochs),
                  "Training loss: {:.6f}".format(avg_cost),
                  "Accuracy: %.4f" % avg_acc,
                  "Learning rate: %.5f" % l_rate)

            # Write summary
            summary = sess.run(merged, feed_dict={cost_ph: avg_cost, acc_ph: avg_acc})
            train_writer.add_summary(summary, e)

        if e % eval_step == 0:
            sess.run(validation_init_op)
            val_avg_cost = 0
            val_avg_acc = 0

            correct_preds = 0

            fights, stat, label, m, _ = sess.run(next_element)

            nb_of_models = 101

            ensemble_acc = 0
            ensemble_cost = 0
            ensemble_preds = []

            for ii in range(nb_of_models):
                model_batch_cost, model_preds, model_preds_cls, val_acc = sess.run([cost, y_pred, y_pred_cls, accuracy],
                                                                  feed_dict={inputs: np.expand_dims(fights, 3),
                                                                             stats: stat,
                                                                             targets: label,
                                                                             target_method: m,
                                                                             keep_prob: prob,
                                                                             is_training: 0})
                ensemble_acc += val_acc
                ensemble_cost += model_batch_cost
                ensemble_preds.append(model_preds_cls)

            val_avg_cost += (ensemble_cost / nb_of_models)
            val_avg_acc += (ensemble_acc / nb_of_models)

            val_loss_curve.append(val_avg_cost)
            val_acc_curve.append(val_avg_acc)

            ensemble_preds = np.divide(np.sum(ensemble_preds, axis=0), (nb_of_models))

            filter = np.where(np.abs(np.round(ensemble_preds) - ensemble_preds) < threshold)

            ensemble_preds_cls = np.round(ensemble_preds)

            total_correct_preds = np.sum(np.equal(np.argmax(label, axis=1), ensemble_preds_cls))

            correct_preds = total_correct_preds

            print()
            print('##########################\n')
            # try:
            #     if filter[0].shape[0] > 50:
            #         val_acc = float(correct_preds) / filter[0].shape[0]
            #     else:
            #         print('Not enough sample for a reliable accuracy')
            #         val_acc = 0
            # except ZeroDivisionError:
            #     val_acc = 0
            print('Threshold: %.2f' % threshold,
                  'Correct predictions: %d' % correct_preds)
            print("Validation loss: {:.6f}".format(val_avg_cost),
                  "Accuracy: %.4f" % (correct_preds / ensemble_preds_cls.shape[0]),
                  "Filter length: %d" % filter[0].shape[0])
            if correct_preds/len(ensemble_preds_cls) > best_avg_acc:
                save_path = saver.save(sess, "pred_saves/model")
                best_avg_acc = correct_preds/len(ensemble_preds_cls)
                counter = 0
            else:
                counter += 1
                print('Current best is %.4f, Incrementing counter to %d' % (best_avg_acc, counter))
            print()
            print('##########################\n')

        # if counter >= 20:
        #     print('Training end, best acc is %.6f' % best_avg_acc)
        #     break

    # Test
    sess.run(testing_init_op)
    saver_test = tf.train.import_meta_graph('pred_saves/model.meta')
    saver_test.restore(sess, tf.train.latest_checkpoint('pred_saves/'))

    graph = tf.get_default_graph()
    test_acc_op = graph.get_tensor_by_name('acc:0')

    nb_test_models = 100

    t_inputs, t_stats, t_targets, t_target_method = graph.get_tensor_by_name('in:0'), \
                                                    graph.get_tensor_by_name('stats:0'), \
                                                    graph.get_tensor_by_name('targ:0'), \
                                                    graph.get_tensor_by_name('target_method:0')

    t_lr = graph.get_tensor_by_name('lr:0')
    t_training = graph.get_tensor_by_name('training:0')

    fights, stat, label, m, _ = sess.run(next_element)

    ensemble_preds = []

    for _ in range(nb_test_models):
        model_preds = sess.run(y_pred_cls, feed_dict={t_inputs: np.expand_dims(fights, 3),
                                                      t_stats: stat,
                                                      t_targets: label,
                                                      t_target_method: m,
                                                      t_lr: l_rate,
                                                      keep_prob: 1.0,
                                                      t_training: 0})
        ensemble_preds.append(model_preds)

    ensemble_preds = np.divide(np.sum(ensemble_preds, axis=0), (nb_test_models))

    filter = np.where(np.abs(np.round(ensemble_preds) - ensemble_preds) < threshold)

    ensemble_preds_cls = np.round(ensemble_preds)

    total_correct_preds = np.sum(np.equal(np.argmax(label, axis=1), ensemble_preds_cls))

    correct_preds = total_correct_preds

    test_acc = correct_preds / len(ensemble_preds_cls)

    # try:
    #     if filter[0].shape[0] > 50:
    #         test_acc = float(correct_preds) / correct_preds/len(ensemble_preds_cls)
    #     else:
    #         print('Not enough sample for a reliable accuracy')
    #         test_acc = 0
    # except ZeroDivisionError:
    #     test_acc = 0

    # with open('results/threshold.txt', 'a') as f:
    #     f.write('Accuracy for threshold of %f is %f\n' % (1 - threshold, test_acc))

    sess.close()

    return best_avg_acc, test_acc, train_loss_curve, val_loss_curve, train_acc_curve, val_acc_curve


if __name__ == '__main__':
    start = time.time()

    test_acc_list, val_acc_list = [], []

    today = datetime.today().strftime('%Y-%m-%d-%H')
    exp_dir = 'results/' + today

    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir)

    for jj in range(1):
        # final_t_loss_curve, final_v_loss_curve, final_t_acc_curve, final_v_acc_curve = [], [], [], []
        avg_t_loss_curve, avg_v_loss_curve, avg_t_acc_curve, avg_v_acc_curve = [], [], [], []

        val_acc, test_acc, t_loss_curve, v_loss_curve, t_acc_curve, v_acc_curve = train(0.5, l, jj, exp_dir)

        test_acc_list.append(test_acc)
        val_acc_list.append(val_acc)

        avg_t_loss_curve.append(t_loss_curve)
        avg_v_loss_curve.append(v_loss_curve)
        avg_t_acc_curve.append(t_acc_curve)
        avg_v_acc_curve.append(v_acc_curve)

        test_acc_avg = np.mean(test_acc_list)
        test_acc_std = np.std(test_acc_list)

        val_acc_avg = np.mean(val_acc_list)
        val_acc_std = np.std(val_acc_list)

        # final_t_loss_curve.append(np.mean(avg_t_loss_curve, axis=1))
        # final_v_loss_curve.append(np.mean(avg_v_loss_curve, axis=1))
        # final_t_acc_curve.append(np.mean(avg_t_acc_curve, axis=1))
        # final_v_acc_curve.append(np.mean(avg_v_acc_curve, axis=1))

        if not os.path.exists('results/lr_search/' + str(l)):
            os.makedirs('results/lr_search/' + str(l))
        else:
            shutil.rmtree('results/lr_search/' + str(l))  # removes all the subdirectories!
            os.makedirs('results/lr_search/' + str(l))

        x_ticks = [i*20 for i, _ in enumerate(np.mean(avg_v_loss_curve, axis=0))]
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.plot(np.mean(avg_t_loss_curve, axis=0), label='training')
        plt.plot(x_ticks, np.mean(avg_v_loss_curve, axis=0), label='validation')
        plt.legend()
        plt.savefig('results/lr_search/' + str(l) + '/loss.png')
        plt.clf()

        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.plot(np.mean(avg_t_acc_curve, axis=0), label='training')
        plt.plot(x_ticks, np.mean(avg_v_acc_curve, axis=0), label='validation')
        plt.legend()
        plt.savefig('results/lr_search/' + str(l) + '/accuracy.png')
        plt.clf()

        curve_dict = {}
        curve_dict['t_acc'] = np.mean(avg_t_acc_curve, axis=0).tolist()
        curve_dict['t_loss'] = np.mean(avg_t_loss_curve, axis=0).tolist()
        curve_dict['v_acc'] = np.mean(avg_v_acc_curve, axis=0).tolist()
        curve_dict['v_loss'] = np.mean(avg_v_loss_curve, axis=0).tolist()

        with open('results/lr_search/' + str(l) + '/data.json', 'w') as f:
            json.dump(curve_dict, f, indent=1)

    end = time.time()
    print('Time to train: %s' % str(end - start))

    with open('results/threshold.txt', 'a') as f:
        f.write('Val acc:\n')
        f.write(str(val_acc_list) + '\n')
        f.write('Mean: %.4f \n' % np.mean(val_acc_list))
        f.write('Test acc:\n')
        f.write(str(test_acc_list) + '\n')
        f.write('Mean: %.4f\n' % np.mean(test_acc_list))

    if sess:
        sess.close()


    # simulate_fights([("Felipe Arantes", "Song Yadong"),
    #                  ("Rolando Dy", "Shane Young"),
    #                  ("Shinsho Anzai", "Jake Matthews"),
    #                  ("Matt Schnell", "Naoki Inoue"),
    #                  ("Jenel Lausa", "Ulka Sasaki")])