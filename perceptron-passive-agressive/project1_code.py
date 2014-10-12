from string import punctuation, digits
import numpy as np
import matplotlib.pyplot as plt
import math

def extract_words(input_string):
    """
      Returns a list of lowercase words in a strong.
      Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')  #replace all characters by _char_

    return input_string.lower().split() #returns a list of all the characters

def extract_dictionary(file):
    """
      Given a text file, returns a dictionary of unique words.
      Each line is passed into extract_words, and a list on unique
      words is maintained. 
    """
    dict = []
    
    f = open(file, 'r')
    for line in f:
        flist = extract_words(line)
        
        for word in flist:
            if(word not in dict):
                dict.append(word) #goes thru each line, each word, adds to list if not in list


    f.close()

    return dict #returns list of unique words

def extract_feature_vectors(file, dict): #makes feature matrix, m tweets x n 0s/1s
    """
      Returns a bag-of-words representation of a text file, given a dictionary.
      The returned matrix is of shape (m, n), where the text file has m non-blank
      lines, and the dictionary has n entries. 
    """
    f = open(file, 'r')
    num_lines = 0  #= number of tweets
    
    for line in f:   #for tweet in file
        if(line.strip()):  #if not blank 
            num_lines = num_lines + 1 #+1 to counter

    f.close()

    feature_matrix = np.zeros([num_lines, len(dict)])  #creates n= # tweets x d= len word list

    f = open(file, 'r')
    pos = 0 #corresponds to ith row tweet f.v.
    
    for line in f:  #for each tweet
        if(line.strip()):
            flist = extract_words(line)  #for each word in that tweet
            for word in flist:
                if(word in dict):  #if in master dList
                    feature_matrix[pos, dict.index(word)] += 1    #UPDATED FOR 6
            pos = pos + 1 #look at next i tweet
            #make (i,j) of (tweet pos, word index)=1 in f.m.
            #already 0 o.w.
    f.close()
    
    return feature_matrix

def averager(feature_matrix, labels):
    """
      Implements a very simple classifier that averages the feature vectors multiplied by the labels.
      Inputs are an (m, n) matrix (m data points and n features) and a length m label vector. 
      Returns a length-n theta vector (theta_0 is 0). 
    """
    (nsamples, nfeatures) = feature_matrix.shape # n tweets x m f.v. words
    theta_vector = np.zeros([nfeatures]) #makes theta vector with theta=0 for each tweet
    theta_0_scalar= 0 
    for i in xrange(0, nsamples): #go through for every tweet
        label = labels[i]  #get that tweet's label (1 or -1)
        sample_vector = feature_matrix[i, :]  #that tweets f.v.
        theta_vector = theta_vector + label*sample_vector #adds corresponding indexes of vectors
        theta_0_scalar += label

    return (theta_vector,theta_0_scalar)

def perceptron_algorithm(feature_matrix, labels):
    nsamples, nfeatures = feature_matrix.shape # n tweets x m words in dict
    theta_vector = np.zeros([nfeatures]) #makes theta vector with theta=0 for every dimension
    theta_0_scalar= 0
    cycles=0

    
    properly_classified=0
    for x in range(0,100): #cycles through data at most 100 times
        #cycles+=1
        for i in xrange(0, nsamples): #go through for every tweet
            
            label = labels[i]  #get that tweet's label (1 or -1)
            sample_vector = feature_matrix[i, :]  #that tweet's f.v.
            if label*(np.dot(theta_vector,sample_vector)+theta_0_scalar)<=0:  #if classifier mis-classifies tweet
                theta_vector += label*sample_vector #adds corresponding indexes of vectors
                theta_0_scalar += label
              
            else:
                properly_classified+=1
                if properly_classified== nsamples:  #correctly classified all samples, return thetas
                    #print cycles
                    return (theta_vector,theta_0_scalar)
        
        properly_classified=0     #reset after full cycle, need to get all right in single cycle   
    return (theta_vector,theta_0_scalar)

def passive_aggressive(feature_matrix, labels):
    nsamples, nfeatures = feature_matrix.shape # n tweets x m words in dict
    theta_vector = np.zeros([nfeatures]) #makes theta vector with theta=0 for every dimension
    hinge_loss=0
    step_size=0
    properly_classified=0
    cycles=0
    for x in range(0,100): #cycles through data at most 100 times
        for i in xrange(0, nsamples): #go through for every tweet
            
            sample_vector = feature_matrix[i, :]  #that tweet's f.v.
            label = labels[i]  #get that tweet's label (1 or -1)
            agreement= label*(np.dot(theta_vector,sample_vector))
            
            if agreement <= 1:
                hinge_loss= 1- agreement
            else:
                hinge_loss=0
                properly_classified+=1
                if properly_classified>nsamples:
                    #print cycles
                    return (theta_vector,0)
                
                
            if hinge_loss!=0:
                step_size= hinge_loss/sum(np.square(sample_vector))
                theta_vector+= step_size*label*sample_vector
        #cycles+=1
        properly_classified=0
    return (theta_vector,0)




def read_vector_file(fname):
    """
      Reads and returns a vector from a file. 
    """
    return np.genfromtxt(fname)

def perceptron_classify(feature_matrix, theta_0, theta_vector):
    """
      Classifies a set of data points given a weight vector and offset.
      Inputs are an (m, n) matrix of input vectors (m data points and n features),
      a real number offset, and a length n parameter vector.
      Returns a length m label vector. 
    """
    (nsamples, nfeatures) = feature_matrix.shape  #nsamples= # of tweets, nfeatures= length of ft. vectors
    label_output = np.zeros([nsamples])  #mleake output vector size of ft. vector length all zeros
    
    for i in xrange(0, nsamples):
        sample_features = feature_matrix[i, :]  #whole feature vector for ith tweet (length d)
        perceptron_output = theta_0 + np.dot(theta_vector, sample_features) #new theta = theta_vector dot tweets feature vector

        if perceptron_output> 0:
            label_output[i] = 1   #if properly classified, label 1
        else:
            label_output[i] = -1   #if on wrong side, label 0

    return label_output  

def write_label_answer(vec, outfile):
    """
      Outputs your label vector the a given file.
      The vector must be of shape (70, ) or (70, 1),
      i.e., 70 rows, or 70 rows and 1 column.
    """
    
    if(vec.shape[0] != 70):
        print("Error - output vector should have 70 rows.")
        print("Aborting write.")
        return

    for v in vec:
        if((v != -1.0) and (v != 1.0)):
            print("Invalid value in input vector.")
            print("Aborting write.")
            return
        
    np.savetxt(outfile, vec)  #makes into gzip file outfile of vector vec
        

def plot_2d_examples(feature_matrix, labels, theta_0, theta):
    """
      Uses Matplotlib to plot a set of labeled instances, and
      a decision boundary line.
      Inputs: an (m, 2) feature_matrix (m data points each with
      2 features), a length-m label vector, and hyper-plane
      parameters theta_0 and length-2 vector theta. 
    """
    
    cols = []
    xs = []
    ys = []
    
    for i in xrange(0, len(labels)):
        if(labels[i] == 1):
            cols.append('b')
        else:
            cols.append('r')
        xs.append(feature_matrix[i][0])
        ys.append(feature_matrix[i][1])

    plt.scatter(xs, ys, s=40, c=cols)

    [xmin, xmax, ymin, ymax] = plt.axis()

    linex = []
    liney = []
    for x in np.linspace(xmin, xmax):
        linex.append(x)
        if(theta[1] != 0.0):
            y = (-theta_0 - theta[0]*x) / (theta[1])
            liney.append(y)
        else:
            liney.append(0)

    plt.plot(linex, liney, 'k-')

    plt.show()

def cross_validation(k):
    '''Takes in K.  Splits the matrix into
fractions, redefines new testing matrix that has deleted 1/fraction columns.
Trains on testing matrix to produce thetas and tests thetas on deleted 1/fraction
columns.  Records error.  Repeats iterations, alternating deleted section.
returns averaged error.'''
    
    dictionary = extract_dictionary('train-tweet.txt')
    labels = read_vector_file('train-answer.txt')
    feature_matrix = extract_feature_vectors('train-tweet.txt', dictionary)

    total_tweets=np.shape(feature_matrix)[0]    
    error=0.0
    size=math.floor(1.0*total_tweets/k)
    tweets_tested=0
    for i in range(0,k):
        #split up feature matrix to before and after the testing fraction.
        #pull out these tweets labels, so indeces still match
        train_matrix=feature_matrix[0:i*size,:]
        train_matrix=np.append(train_matrix, feature_matrix[(i+1)*size:-1],axis=0)
        train_labels= np.append(labels[0:i*size],labels[(i+1)*size:-1],axis=1)

        #pull out testing matrix and corresponding labels    
        test_matrix=feature_matrix[i*size:(i+1)*size,:]
        test_labels=labels[i*size:(i+1)*size]

        average_theta,average_theta_0 = perceptron_algorithm(train_matrix, train_labels)
        test_label_out = perceptron_classify(test_matrix, average_theta_0, average_theta) #returns labels vector
        tweets_tested+=1 #will be total # of tweets tested
        if test_label_out[i] != test_labels[i]: #check test label w actual, up errors by one if mismatch
            error+=1
            
          

    return 1.0*error/tweets_tested

    
    
    
