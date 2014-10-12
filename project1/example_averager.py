import numpy as np
import project1_code as p1  #################

dictionary = p1.extract_dictionary('train-tweet.txt')
labels = p1.read_vector_file('train-answer.txt')
feature_matrix = p1.extract_feature_vectors('train-tweet.txt', dictionary)

#Using Averager w Offset
average_theta,average_theta_0 = p1.averager(feature_matrix, labels)
label_output = p1.perceptron_classify(feature_matrix, average_theta_0, average_theta)

correct = 0
for i in xrange(0, len(label_output)):
    if(label_output[i] == labels[i]):
        correct = correct + 1

percentage_correct = 100.0 * correct / len(label_output)
print("Averager gets " + str(percentage_correct) + "% correct (" + str(correct) + " out of " + str(len(label_output)) + ").")



def example(algorithm):
    
    dictionary = p1.extract_dictionary('train-tweet.txt')
    labels = p1.read_vector_file('train-answer.txt')
    feature_matrix = p1.extract_feature_vectors('train-tweet.txt', dictionary)
    
    if algorithm== 'averager':
        average_theta,average_theta_0 = p1.averager(feature_matrix, labels)
    elif algorithm== 'perceptron':
        average_theta,average_theta_0 = p1.perceptron_algorithm(feature_matrix, labels)
    elif algorithm== 'passive':
        average_theta,average_theta_0 = p1.passive_aggressive(feature_matrix, labels)

    

    label_output = p1.perceptron_classify(feature_matrix, average_theta_0, average_theta)

    correct = 0
    for i in xrange(0, len(label_output)):
        if(label_output[i] == labels[i]):
            correct +=1

    percentage_correct = 100.0 * correct / len(label_output)
    print(algorithm + " gets " + str(percentage_correct) + "% correct (" + str(correct) + " out of " + str(len(label_output)) + ").")

##example('averager')
##example('perceptron')
##example('passive')


di = p1.extract_dictionary('train-tweet.txt')
labels = p1.read_vector_file('train-answer.txt')
fm= p1.extract_feature_vectors('test-tweet.txt', di)
th1,th2=p1.perceptron_algorithm(fm,labels)
lbls=p1.perceptron_classify(fm,th2,th1)
p1.write_label_answer(lbls, 'tweet_labels.txt')





