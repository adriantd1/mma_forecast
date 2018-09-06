import os
import sqlite3
from pprint import pprint

import numpy as np
from collections import defaultdict

import random

import fightmetric as fm
import fmcrawler_sql as fmc
import tensorflow as tf

from fmcrawler_sql import get_fighter_stats


def simulated_fight_data(f1_name, f2_name, dbfile='mma_db.sqlite'):
    # conn = sqlite3.connect(dbfile, timeout=10)
    #
    # cur = conn.cursor()
    #
    # fighter_dict = {f1_name: [], f2_name: []}
    #
    # for name in fighter_dict.keys():
    #     cur.execute("SELECT * FROM Fights where fighter1=name or fighter2=f2_name ORDER BY date DESC")
    #
    #     fights = cur.fetchall()

    conn = sqlite3.connect(dbfile, timeout=10)

    cur = conn.cursor()

    cur.execute('SELECT id, fighter1, fighter2, method, winner '
                'from Fights '
                'where fighter1= ? or fighter2= ? or fighter1= ? or fighter2= ? '
                'order by date', (f1_name, f1_name, f2_name, f2_name))

    fights = cur.fetchall()

    fighter_dict = defaultdict(list)

    methods = set()

    for fight in fights:
        cur.execute('SELECT * FROM Round WHERE fightid=' + str(fight[0]) + ' ORDER BY round_number')
        rounds = cur.fetchall()
        f1, f2 = defaultdict(list), defaultdict(list)
        done = []

        fighter1, fighter2 = '', ''
        method = fight[3]
        if 'KO' in method:
            method = 'KO'
        elif 'SUB' in method:
            method = 'SUB'
        methods.add(method)
        winner = fight[4]
        for r in rounds:
            fighter1, fighter2 = r[3], r[4]
            if r[1] in done:
                continue
            done.append(r[1])
            l1, l2 = [], []
            for i in range(5, len(r)):
                if i % 2 == 1:
                    l1.append(r[i])
                else:
                    l2.append(r[i])

            total_1 = np.subtract(l1, l2)
            total_2 = np.multiply(-1, total_1)

            if total_1 != []:
                f1[fighter1].append(total_1.tolist())
                f2[fighter2].append(total_2.tolist())

        if len(list(f1.keys())) != 0:
            while len(f1[fighter1]) < 5:
                f1[fighter1].append([0] * 20)
                f2[fighter2].append([0] * 20)
        if len(f1[fighter1]) == 5 and len(f2[fighter2]) == 5:
            fighter_dict[fighter1].append((np.asarray(f1[fighter1], dtype=np.float32), winner, method, fighter2))
            fighter_dict[fighter2].append((np.asarray(f2[fighter2], dtype=np.float32), winner, method, fighter1))
        else:
            print('Defective data fight %d' % fight[0])

    for k, v in fighter_dict.items():
        fighter_dict[k] = np.asarray(v)

    fighters = [f1_name, f2_name]
    final_dict = {k:v for k,v in fighter_dict.items() if k in fighters}

    return final_dict


def average_predictions(arr):
    """
    Take N predictions of M instances and return an array of length N with the most common prediction
    :param arr:
    :return:
    """
    pass


def new_weights(shape, name='weights'):
    initializer = tf.contrib.layers.xavier_initializer()
    w = tf.Variable(initializer(shape), name=name)
    tf.summary.histogram('weight_hist', w)
    return w


def new_biases(length, name='bias'):
    b = tf.Variable(tf.constant(0.05, shape=[length]), name=name)
    tf.summary.histogram('bias_hist', b)
    return b


def new_fc_layer(input,  # The previous layer.
                 num_inputs,  # Num. inputs from prev. layer.
                 num_outputs,  # Num. outputs.
                 use_elu=True,
                 use_dropout=True,
                 keep_prob=0.5,
                 layer_name='fc'):  # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs], name=layer_name + '/weights')
    biases = new_biases(length=num_outputs, name=layer_name + '/bias')

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    if use_elu:
        layer = tf.nn.relu(layer)

    if use_dropout:
        dropout = tf.nn.dropout(layer, keep_prob)
        return dropout

    return layer


def format_data(data, simulating=False):
    '''
    Build input from dict. Use previous 5 fights info to predict the outcome of the next fight. If fighter has less than
    5 fights, pad with zeros
    :param data:
    :return:
    '''
    # stats = get_fighter_stats()

    formatted_data = []

    lst= list(data.items())

    random.shuffle(lst)

    for k,v in lst:
        if len(v) >= 6:
            for i in range(5, len(v)):
                input = {'fight_arr': {'fights': [], 'fights_stats': []}}
                for j in range(i-5, i):
                    input['fight_arr']['fights'].append(np.asarray(v[j]['fight']))
                    input['fight_arr']['fights_stats'].append(np.asarray(v[j]['fight_stats']))

                if v[i]['winner'] == k:
                    winner = (0, 1)
                else:
                    winner = (1, 0)
                input['winner'] = winner
                input['method'] = label_for_method(v[i]['method'])
                input['date'] = v[i]['date']
                input['fighter'] = k
                input['fighter_stats'] = v[i]['fighter_stats']

                input['cumul_stats'] = cum_stat_from_data(input)

                if np.asarray(input['fight_arr']['fights']).shape != (5, 5, 2, 22):
                    print('Bad shape')
                    print(np.asarray(input['fight_arr']['fights']).shape)
                    continue

                # # Build stats
                # input['stats'] = []
                #
                # # Height, weight, stance
                # input['stats'].append(stats[k][0])
                # input['stats'].append(stats[k][1])
                # input['stats'].extend(stats[k][3:8])
                #
                # # Total strikes
                # val = 0
                # for fight in input['fight']:
                #     for round in fight:
                #         val += (round[0][0] - round[0][1])
                # input['stats'].append(val)
                #
                # # Total strikes pct
                # pct = 0
                # total = 0
                # for fight in input['fight']:
                #     for round in fight:
                #         pct += round[0][0] * round[0][1]
                #         total += round[0][0]
                # input['stats'].append(pct/total)

                formatted_data.append(input)

    # ids_to_del = set()
    # for k, v in data.items():
    #     if len(v) >= 6:
    #         # data[k] = v[-6:]
    #         # assert len(data[k]) == 5
    #         continue
    #     elif len(v) <= 5 and not simulating:
    #         ids_to_del.add(k)
    #     else:
    #         temp = list(v)
    #         while len(temp) < 6:
    #             temp.insert(0, (np.zeros((5, 20)), 'None', 'None', 'None'))
    #         data[k] = np.asarray(temp)
    #
    # for k in ids_to_del:
    #     del data[k]

    # return data

    return formatted_data


def transform_data(data, sess):
    '''
    Transform data by passing each fight through the autoencoder.
    :param data:
    :return:
    '''
    # sess.run(valid_iterator.initializer)

    saver = tf.train.import_meta_graph('saves/model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('saves/'))

    graph = tf.get_default_graph()
    decoded_img = graph.get_tensor_by_name('decoded:0')

    last_conv = graph.get_tensor_by_name('encoded:0')

    inp = graph.get_tensor_by_name('inputs:0')

    # for k, v in data.items():
    #     for i in range(len(v)):
    #         # print(np.asarray(np.expand_dims(np.expand_dims(v[i], 2), 0)).shape)
    #         fight = sess.run(last_conv, feed_dict={inp: np.asarray(np.expand_dims(np.expand_dims(v[i][0], 2), 0))})
    #         v[i] = (np.squeeze(fight).flatten(), v[i][1], v[i][2], v[i][3])

    for i in range(len(data)):
        for f in range(len(data[i]['fight_arr']['fights'])):
            try:
                fight = sess.run(last_conv, feed_dict={inp: np.expand_dims(np.expand_dims(data[i]['fight_arr']['fights'][f], 3), 0)})
                data[i]['fight_arr']['fights'][f] = np.asarray(fight).flatten()
            except:
                continue


    return data


def cum_stat_from_data(dct):
    stat_arr = []
    fighter = dct['fighter']

    # pprint(dct['fight_arr'])

    fight_cumul = np.zeros((2, 22))

    for i, f in enumerate(dct['fight_arr']['fights']):
        for r in f:
            fight_cumul += r[0]

    # Total fight time
    total_min = 0
    for fight_stats in dct['fight_arr']['fights_stats']:
        # print(fight_stats)
        total_min += float(fight_stats[15])
    total_min = total_min/60.0

    # Total sig strike landed / min
    total_strike = 0
    total_opp_strike = 0
    for fight_stats in dct['fight_arr']['fights_stats']:
        if fight_stats[1] == fighter:
            total_strike += float(fight_stats[9])
            total_opp_strike += float(fight_stats[10])
        else:
            total_strike += float(fight_stats[10])
            total_opp_strike += float(fight_stats[9])
    stat_arr.append((total_strike / total_min))

    # Total sig strike head landed / min
    total_strike_head = fight_cumul[0][2]
    stat_arr.append(total_strike_head / total_min)

    # Total sig strike body landed / min
    total_strike_body = fight_cumul[0][4]
    stat_arr.append(total_strike_body / total_min)

    # Total sig strike leg landed / min
    total_strike_leg = fight_cumul[0][6]
    stat_arr.append(total_strike_leg / total_min)

    # Total sig strike from distance landed / min
    total_strike_distance = fight_cumul[0][8]
    stat_arr.append(total_strike_distance / total_min)

    # Total sig strike from clinch landed / min
    total_strike_clinch = fight_cumul[0][10]
    stat_arr.append(total_strike_clinch / total_min)

    # Total sig strike from ground landed / min
    total_strike_ground = fight_cumul[0][12]
    stat_arr.append(total_strike_ground / total_min)

    # Pct total strike landed
    total_strikes = 0
    total_attempts = 0
    for fight in dct['fight_arr']['fights']:
        for r in fight:
            total_strikes += r[0][0]
            if r[0][1] == 0:
                total_attempts += r[0][0]
            else:
                total_attempts += r[0][0] / r[0][1]
    stat_arr.append(total_strikes/total_attempts)

    # Knockdowns
    kd = 0
    opp_kd = 0
    for fight_stats in dct['fight_arr']['fights_stats']:
        if 'KO' in fight_stats[5]:
            if fight_stats[16] == fighter:
                kd += 1
            else:
                opp_kd += 1
    try:
        stat_arr.append((kd - opp_kd) / kd)
    except ZeroDivisionError:
        stat_arr.append(-opp_kd)

    # Takedown
    td = 0
    for fight_stats in dct['fight_arr']['fights_stats']:
        if fight_stats[1] == fighter:
            td += float(fight_stats[13])
        else:
            total_strike += float(fight_stats[14])
    stat_arr.append(td * 15 / total_min)

    # Submission attempt
    sub = 0
    opp_sub = 0
    for fight_stats in dct['fight_arr']['fights_stats']:
        if fight_stats[1] == fighter:
            sub += float(fight_stats[11])
            opp_sub += float(fight_stats[12])
        else:
            sub += float(fight_stats[12])
            opp_sub += float(fight_stats[11])
    try:
        stat_arr.append((sub - opp_sub) / sub)
    except ZeroDivisionError:
        stat_arr.append(-opp_sub)

    # Str ratio
    try:
        stat_arr.append(total_strike/total_opp_strike)
    except ZeroDivisionError:
        stat_arr.append(total_strike)

    # Log str ratio
    try:
        stat_arr.append(np.log(total_strike / total_opp_strike))
    except ZeroDivisionError:
        stat_arr.append(np.log(total_strike))

    # Str diff per min
    stat_arr.append((total_strike - total_opp_strike)/total_min)

    stat_arr.extend(dct['fighter_stats'])

    return stat_arr


def get_data_with_labels(data):
    stats = get_fighter_stats()

    sheet, s, l, m = [], [], [], []
    for k, v in data.items():
        for i in range(5, len(v)):
            fighter_sheet = []
            for f in v[i - 5:i]:
                fighter_sheet.append(f[0])
            f_stat = stats.get(k, None)
            opp_stat = stats.get(v[i][3], None)
            if f_stat is None or opp_stat is None:
                print('Skiping fighter %s' % k)
                continue
            stat = np.subtract(f_stat, opp_stat)
            winner = v[i][1]
            method = v[i][2]
            # input.append((fighter_sheet, stat, label))
            if winner == k:
                label = (0., 1.)
            else:
                label = (1., 0.)

            sheet.append(fighter_sheet)
            s.append(stat)
            l.append(label)
            m.append(label_for_method(method))

    return np.asarray(sheet, dtype=np.float32), np.asarray(s), np.asarray(l), np.asarray(m)


def augment_data(input_sheet, stats, label, method, centroids, augmentation_factor=2):
    augmented_sheet, augmented_stats, augmented_label, augmented_method, augmented_centroids = [], [], [], [], []
    for i in range(len(input_sheet)):
        for _ in range(augmentation_factor-1):
            noisy_fighter_sheet = []
            for j in range(len(input_sheet[i])):
                noisy_fighter_sheet.append(np.random.normal(0, 0.0225, size=input_sheet[i][j].shape) + input_sheet[i][j])
            noisy_stat = np.random.normal(0, 0.0225, stats[i].shape) + stats[i]

            augmented_sheet.append(noisy_fighter_sheet)
            augmented_stats.append(noisy_stat)
            augmented_label.append(label[i])
            augmented_method.append(method[i])
            augmented_centroids.append(centroids[i])

    return np.concatenate((input_sheet, augmented_sheet), axis=0), \
           np.concatenate((stats, augmented_stats), axis=0),\
           np.concatenate((label, augmented_label), axis=0), \
           np.concatenate((method, augmented_method), axis=0), \
           np.concatenate((centroids, augmented_centroids), axis=0)


def augment_rnn_data(input_sheet, stats, label, method, augmentation_factor=2):
    augmented_input, augmented_stats, augmented_label, augmented_method = [], [], [], []
    for i in range(len(input_sheet)):
        for _ in range(augmentation_factor - 1):
            noisy_fighter_sheet = []
            for j in range(len(input_sheet[i])):
                noisy_fighter_sheet.append(np.random.normal(0, 0.01, size=input_sheet[i][j].shape) + input_sheet[i][j])
            noisy_stat = np.random.normal(0, 0.01, stats[i].shape) + stats[i]

            augmented_input.append(noisy_fighter_sheet)
            augmented_stats.append(noisy_stat)
            augmented_label.append(label[i])
            augmented_method.append(method[i])

    return np.concatenate((input_sheet, augmented_input), axis=0).astype(np.float32), \
           np.concatenate((stats, augmented_stats)), \
           np.concatenate((label, augmented_label), axis=0), \
           np.concatenate((method, augmented_method), axis=0)



def label_for_method(method):
    vector = []
    if method == 'M-DEC':
        vector = [1, 0, 0, 0, 0, 0, 0, 0, 0]
    elif method == 'Other':
        vector = [0, 1, 0, 0, 0, 0, 0, 0, 0]
    elif method == 'SUB':
        vector = [0, 0, 1, 0, 0, 0, 0, 0, 0]
    elif method == 'U-DEC':
        vector = [0, 0, 0, 1, 0, 0, 0, 0, 0]
    elif method == 'DQ':
        vector = [0, 0, 0, 0, 1, 0, 0, 0, 0]
    elif method == 'DQRearNakedChoke':
        vector = [0, 0, 0, 0, 0, 1, 0, 0, 0]
    elif method == 'KO':
        vector = [0, 0, 0, 0, 0, 0, 1, 0, 0]
    elif method == 'S-DEC':
        vector = [0, 0, 0, 0, 0, 0, 0, 1, 0]
    elif method == 'Decision':
        vector = [0, 0, 0, 0, 0, 0, 0, 0, 1]

    return vector


def add_centroids(data, sess):
    global cc, count
    # K-means
    clusters_n = 3
    iteration_n = 2000

    points = tf.convert_to_tensor(data[1])
    centroids = tf.Variable(tf.slice(tf.random_shuffle(points), [0, 0], [clusters_n, -1]))

    print('Points dim')
    print(points.shape)

    print('Centroids dim')
    print(centroids.shape)

    points_expanded = tf.expand_dims(points, 0)
    centroids_expanded = tf.expand_dims(centroids, 1)

    print('Points dim')
    print(points_expanded.shape)

    print('Centroids dim')
    print(centroids_expanded.shape)

    sub = tf.subtract(points_expanded, centroids_expanded)

    print('sub shape')
    print(sub.shape)

    distances = tf.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 2)
    assignments = tf.argmin(distances, 0)

    means = []
    for c in range(clusters_n):
        means.append(tf.reduce_mean(
            tf.gather(points,
                      tf.reshape(
                          tf.where(
                              tf.equal(assignments, c)
                          ), [1, -1])
                      ), reduction_indices=[1]))

    new_centroids = tf.concat(means, 0)

    update_centroids = tf.assign(centroids, new_centroids)

    # Get centroids
    sess.run(tf.global_variables_initializer())
    for step in range(iteration_n):
        [_, centroid_values, points_values, assignment_values] = sess.run(
            [update_centroids, centroids, points, assignments])

        # cluster_dict = {i:[] for i in range(clusters_n)}

    mean_dist = [0 for _ in range(clusters_n)]
    count = [0 for _ in range(clusters_n)]
    for i in range(len(assignment_values)):
        mean_dist[assignment_values[i]] += np.linalg.norm(centroid_values[assignment_values[i]] - points_values[i])
        count[assignment_values[i]] += 1

    print('Centroids are:')
    print(centroid_values)
    print(count)

    cc = np.argmax(count)

    # closest_centroid = []
    # for i in range(len(data[0])):
    #     closest_centroid.append(np.argmin([np.linalg.norm(data[1][i] - c) for c in centroid_values]))

    print(assignment_values)

    return data[0], data[1], data[2], data[3], assignment_values


if __name__ == '__main__':
    # pprint(simulated_fight_data("Georges St-Pierre", "Josh Koscheck"))
    # fighters = fmc.get_fighters()
    # formatted_fighters = format_data(fighters)
    # pprint(formatted_fighters)
    # print(len(formatted_fighters))

    sess = tf.Session()

    data = transform_data(format_data(fmc.get_fighters()), sess)
    print(data[0])

    sess.close()