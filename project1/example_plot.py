import project1_code as p1

pos1 = [-0.3, 0.4]
pos2 = [0.2, 0.3]
neg1 = [-0.1, -0.1]
neg2 = [0.3, 0.1]

feature_matrix = [pos1, pos2, neg1, neg2]
labels = [1, 1, -1, -1]

p1.plot_2d_examples(feature_matrix, labels, 0, [0.25, 0.6])
