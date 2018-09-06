import tensorflow as tf
from fmcrawler_sql import *

from main import augment_data, get_data_with_labels, transform_data, format_data

input_sheet, fighters_stat, labels, methods = get_data_with_labels(transform_data(format_data(get_fighters())))

permutations = np.random.permutation(input_sheet.shape[0])

input_sheet = input_sheet[permutations]
fighters_stat = fighters_stat[permutations]
labels = labels[permutations]
methods = methods[permutations]

split1 = int(len(permutations) * 0.8)
split2 = int(len(permutations) * 0.9)

train_sheet, train_stats, train_labels, train_methods = augment_data(input_sheet[:split1], fighters_stat[:split1], labels[:split1], methods[:split1], 4)

print(train_sheet.shape, train_stats.shape)

points_n = 200
clusters_n = 6
iteration_n = 2000

points = tf.convert_to_tensor(train_stats)
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
init = tf.initialize_all_variables()

def get_centroids():
    with tf.Session() as sess:
        sess.run(init)
        for step in range(iteration_n):
            [_, centroid_values, points_values, assignment_values] = sess.run(
                [update_centroids, centroids, points, assignments])

            # cluster_dict = {i:[] for i in range(clusters_n)}

        mean_dist = [0 for _ in range(clusters_n)]
        count = [0 for _ in range(clusters_n)]
        for i in range(len(assignment_values)):
            mean_dist[assignment_values[i]] += np.linalg.norm(centroid_values[assignment_values[i]] - points_values[i])
            count[assignment_values[i]] += 1

        for i in range(len(count)):
            mean_dist[i] = mean_dist[i]/count[i]

        return centroid_values




# plt.scatter(points_values[:, 0], points_values[:, 1], c=assignment_values, s=50, alpha=0.5)
# plt.plot(centroid_values[:, 0], centroid_values[:, 1], 'kx', markersize=15)
# plt.show()