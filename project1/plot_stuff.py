import project1_code as p1
import numpy as np

#training
##dictionary = p1.extract_dictionary('train-tweet.txt')
##training_labels = p1.read_vector_file('train-answer.txt')
##training_feature_matrix = p1.extract_feature_vectors('train-tweet.txt', dictionary)
##
##ta, ta0 = p1.averager(training_feature_matrix, training_labels)
##tpc,tpc0= p1.perceptron_algorithm(training_feature_matrix, training_labels)
##tps,tps0= p1.passive_aggressive(training_feature_matrix, training_labels)
##
##
###testing
##testing_feature_matrix = p1.extract_feature_vectors('test-tweet.txt', dictionary)
##
##av_test_labels=p1.perceptron_classify(testing_feature_matrix, ta0, ta)
##pc_test_labels=p1.perceptron_classify(testing_feature_matrix, tpc0, tpc)
##ps_test_labels=p1.perceptron_classify(testing_feature_matrix, tps0, tps)
##
###plotting
####p1.plot_2d_examples(testing_feature_matrix, av_test_labels, ta0, ta)
####p1.plot_2d_examples(testing_feature_matrix, ps_test_labels, tps0, tps)
####p1.plot_2d_examples(testing_feature_matrix, pc_test_labels, tpc0, tpc)

##feature_matrix= np.array([[-3,2],[-1,1],[-1,-1],[2,2],[1,-1]])
##labels=([1,1,-1,-1,-1])
##feature_matrix= np.array([[3,3],[3,4],[3,2],[3,5],[2,1],[-1,0],[1,1],[-1,1],[-1,-1],[4,1],[5,1],[4,1],[1,-2],[4,0],[3,0],[3,-1]])
##labels=([1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1])
feature_matrix= np.array([[0,5],[0,4],[0,6],[1,5],[1,7],[-1,5],[-1,6],[-1,7],[-1,-1],[-4,1],[-5,3],[-4,-3],[-1,-2],[-3,-4],[-2,-8],[0,-5]])
labels=([1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1])
theta, theta_0= p1.averager(feature_matrix, labels)
p1.plot_2d_examples(feature_matrix, labels, theta_0, theta)
