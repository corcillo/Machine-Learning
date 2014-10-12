from sklearn.datasets import fetch_lfw_people
from sklearn.cross_validation import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sklearn
import sklearn.svm
from sklearn.svm import SVC
import random

# Returns
# X: an n x d array, in which each row represents an image
# y: a 1 x n vector, elements of which are integers between 0 and nc-1
#    where nc is the number of classes represented in the data

# Warning: this will take a long time the first time you run it.  It
# will download data onto your disk, but then will use the local copy
# thereafter.  
def getData():
    global X, n, d, y, h, w
    lfw_people = fetch_lfw_people(min_faces_per_person=40, resize=0.4)
    n, h, w = lfw_people.images.shape
    X = lfw_people.data
    d = X.shape[1]
    y = lfw_people.target
    n_classes = lfw_people.target_names.shape[0]
    print("Total dataset size:")
    print("n_samples: %d" % n)
    print("n_features: %d" % d)
    print("n_classes: %d" % n_classes)
    return X, y

# Input
# im: a row or column vector of dimension d
# size: a pair of positive integers (i, j) such that i * j = d
#       defaults to the right value for our images
# Opens a new window and displays the image
lfw_imageSize = (50,37)
def showIm(im, size = lfw_imageSize):
    plt.figure()
    im = im.copy()
    im.resize(*size)
    plt.imshow(im.astype(float), cmap = cm.gray)

# Take an eigenvector and make it into an image
def vecToImage(x, size = lfw_imageSize):
  im = x/np.linalg.norm(x)
  im = im*(256./np.max(im))
  im.resize(*size)
  return im

# Plot an array of images
# Input
# - images: a 12 by d array
# - title: string title for whole window
# - subtitles: a list of 12 strings to be used as subtitles for the
#              subimages, or an empty list
# - h, w, n_row, n_col: can be used for other image sizes or other
#           numbers of images in the gallery

def plotGallery(images, title='plot', subtitles = [],
                 h=50, w=37, n_row=3, n_col=4):
    plt.figure(title,figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(min(len(images), n_row * n_col)):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        if subtitles:
            plt.title(subtitles[i], size=12)
        plt.xticks(())
        plt.yticks(())    
    
# Perform PCA, optionally apply the "sphering" or "whitening" transform, in
# which each eigenvector is scaled by 1/sqrt(lambda) where lambda is
# the associated eigenvalue.  This has the effect of transforming the
# data not just into an axis-aligned ellipse, but into a sphere.  
# Input:
# - X: n by d array representing n d-dimensional data points
# Output:
# - u: d by n array representing n d-dimensional eigenvectors;
#      each column is a unit eigenvector; sorted by eigenvalue
# - mu: 1 by d array representing the mean of the input data
# This version uses SVD for better numerical performance when d >> n

def PCA(X, sphere = False):
    (n, d) = X.shape
    mu = np.mean(X, axis=0)
    (x, l, v) = np.linalg.svd(X-mu)
    l = np.hstack([l, np.zeros(v.shape[0] - l.shape[0], dtype=float)])
    u = np.array([vi/(li if (sphere and li>1.0e-10) else 1.0) \
                  for (li, vi) \
                  in sorted(zip(l, v), reverse=True, key=lambda x: x[0])]).T
    return u, mu

# Selects a subset of images from the large data set.  User can
# specify desired classes and desired number of images per class.
# Input:
# - X: n by d array representing n d-dimensional data points
# - y: 1 by n array reprsenting the integer class labels of the data points
# - classes: a list of integers naming a subset of the classes in the data
# - nim: number of integers desired
# Return:
# - X1: nim * len(classes) by d array of images
# - y1: 1 by nim * len(classes) array of class labels
def limitPics(X, y, classes, nim):
  (n, d) = X.shape
  k = len(classes)
  X1 = np.zeros((k*nim, d), dtype=float)
  y1 = np.zeros(k*nim, dtype = int)
  index = 0
  for ni, i in enumerate(classes):      # for each class
    count = 0                           # count how many samples in class so far
    for j in range(n):                  # look over the data
      if count < nim and y[j] == i:     # element of class
        X1[index] = X[j]
        y1[index] = ni
        index += 1
        count += 1
  return X1, y1

# Provides an initial set of data points to use to initialize
# clustering.  It "cheats" by using the class labels, and picks the
# medoid of each class.
# Input:
# - X: n by d array representing n d-dimensional data points
# - y: 1 by n array representing integer class labels
# - k: number of classes
# Output:
# - init: k by d array representing initial cluster medoids
def cheatInit(X, y, k):
    (n, d) = X.shape
    init = np.zeros((k, d), dtype=float)
    for i in range(k):
        (index, dist) = cheatIndex(X, y, i, l2Sq)
        init[i] = X[index]
    return init

def l2Sq (x,y):
    return np.sum(np.dot((x-y), (x-y).T))

def cheatIndex(X, clusters, j, metric):
    n, d = X.shape
    bestDist = 1.0e10
    index = 0
    for i1 in xrange(n):
        if clusters[i1] == j:
            dist = 0
            C = X[i1,:]
            for i2 in xrange(n):
                if clusters[i2]  == j:
                    dist += metric(C, X[i2,:])
            # print dist
            if dist < bestDist:
                bestDist = dist
                index = i1
    return index, bestDist


# Scores the quality of a clustering, in terms of its agreement with a
# vector of labels
# Input:
# - clustering: (medoids, clusters, indices) of type returned from kMedoids
# - y: 1 by n array representing integer class labels
# Output:
# numerical score between 0 and 1
def scoreMedoids(clustering, y):
    (medoids, mIndex, cluster) = clustering
    n = cluster.shape[0]                  # how many samples
    # The actual label for each medoid, which we associate with
    # samples in cluster
    medoidLabels = np.array([y[i] for i in mIndex]) 
    print medoidLabels
    count = len(set(medoidLabels.tolist())) # how many actual people predicted
    # For each sample, what is the label implied by its cluster
    clusterLabels = np.array([medoidLabels[c] for c in cluster])
    score = sum([1 if y[i]==clusterLabels[i] else 0 \
                 for i in xrange(n)])/float(n)
    return score
'''
graphAny will graph any matrix by point, all black
graphAMatrix takes in a matrix, a list of rows to plot from it, and the color index
graphCenters takes in a centroid matrix and plots them with all different colors
'''    
def graphAny(matrix):
    for row in range(0, len(matrix)):
        plt.scatter(matrix[row,0],matrix[row,1], color="black")
        plt.show()    
def graphAMatrix(matrix, listOfRows, Zrow):
    ZColors= ["pink", "purple", "green","orange","yellow","pink","purple","maroon","brown","black","cyan","magenta"]
    for row in listOfRows:
        plt.scatter(matrix[row,0],matrix[row,1], color=ZColors[Zrow])
    plt.show()
def graphCenters(CenterMatrix):
    ZColors= ["red", "blue", "green","orange","yellow","pink","purple","maroon","brown","black","cyan","magenta"]
    for zrow in range(0, len(CenterMatrix)):
        plt.scatter(CenterMatrix[zrow,0],CenterMatrix[zrow,1], color=ZColors[zrow])
    plt.show()
'''
takes in X, an nxd matrix of n data points with d dimensional feat. vectors.
init: a kxd array of k datapoints with d features, initial guesses for centroids

return (centroids, clusterAssignment) where
centroids is a kxd array of k data points with d features, indicating final centroids
clusterAssignment, an array of integers, each between 0 and k-1 
                    indicating which cluster the input points are assigned to.
---Make a 2d data set with three clusters.  
---Apply algorithm to data set with reasonable initial centroids
---producei grapsh showing cluster assignments at each iteration
'''    
def ml_k_means(X, init):
    #helper methods to keep math easy
    numXRows= X.shape[0]       #contain x's
    numInitRows= init.shape[0] #contain z's
    cost=[1,0] #(previous cost, new cost)
    clusters=[[] for x in range(numInitRows)] #k cluster lists
    graphCenters(init)
    graphAny(X)
    while cost[1]<=cost[0]: #while cost decreased    
        cost[0]=cost[1] 
        #1. assign each point to closest center's cluster
        clusters=[[] for x in range(numInitRows)] 
        for rowx in range(numXRows):  
            z=init[0,:]      
            x=X[rowx,:]      
            minDist= (np.linalg.norm(x-z))**2   
            clusterIndex=0 
            for rowz in range(numInitRows):
                z= init[rowz,:]
                if (np.linalg.norm(x-z))**2<minDist:
                    minDist= (np.linalg.norm(x-z))**2 
                    clusterIndex=rowz              
            clusters[clusterIndex].append(rowx)
         
        #for clus in range(len(clusters)):
            #graphAMatrix(X,clusters[clus],clus)
        #2. Reassign each cluster's center to be mean value of cluster
        for clusNum in range(len(clusters)):
            summ=0
            for point in clusters[clusNum]:
                summ+=X[point,:]
              
            if len(clusters[clusNum])!=0:  
                init[clusNum,:]= summ/len(clusters[clusNum])
            else:
                init[clusNum,:]=0
        #graphCenters(init)
            
        #3. caluclate cost based on reassigned x's and avg z's
        newCost=0
        for cluster in clusters:
            for point in cluster:
                for z in init: 
                    newCost+=(np.linalg.norm(X[point,:]-z))**2
        cost[1]=newCost   
    clusterAssignments= [0 for x in range(numXRows)]
    for clusterNum in range(len(clusters)):
         for point in clusters[clusterNum] :
             clusterAssignments[point]=clusterNum 
    print clusters
    for clus in range(len(clusters)):
        graphAMatrix(X,clusters[clus],clus)
    graphCenters(init)
    return(init,np.array(clusterAssignments))

'''
takes in X, an nxd matrix of n data points with d dimensional feat. vectors.
init: a kxd array of k datapoints with d features, initial Centroids, MUST BE DATA POINTS FROM X
return (centroids, clusterAssignment) where
centroids is a kxd array of k data points with d features, indicating final centroids
clusterAssignment, an array of integers, each between 0 and k-1 
                    indicating which cluster the input points are assigned to.

'''    
def kmedoids(X, init):
    #helper methods to keep math easy
    numXRows= X.shape[0]       #contain x's
    numInitRows= init.shape[0] #contain z's
    cost=[1,0] #(previous cost, new cost)
    clusters=[[] for x in range(numInitRows)] #k cluster lists
    graphCenters(init)
    graphAny(X)
    while cost[1]<=cost[0]: #while cost decreased 
        cost[0]=cost[1] 
        
        #1. assign each point to closest center's cluster
        clusters=[[] for x in range(numInitRows)] 
        for rowx in range(numXRows):  
            z=init[0,:]      
            x=X[rowx,:]      
            minDist= (np.linalg.norm(x-z))**2   
            clusterIndex=0 
            for rowz in range(numInitRows):
                z= init[rowz,:]
                if (np.linalg.norm(x-z))**2<minDist:
                    minDist= (np.linalg.norm(x-z))**2 
                    clusterIndex=rowz              
            clusters[clusterIndex].append(rowx)
    
        #2. Reassign each cluster's center to be mean value of cluster
        ##list of clusters, which are lists of x's
        ##for each cluster, center = cluster[0]
        ##minSumDistanceSquareds= distance between center and all other points in list
        newTotalAvgCost=0
        medoidsIndexList= [0 for i in range(len(init))]
        
        #iterate through clusters
        for clusNum in range(len(clusters)):
            centerP=clusters[clusNum][0] #set center of cluster to be first one in cluster
            minDSum=0                   #each cluster has a min total cost
            for point in clusters[clusNum]:  #iterate through points in that cluster
                minDSum+=np.linalg.norm(point-centerP)**2  #minDSum= sum of distances from z to x's
             #iterate through testCenters in this cluster after first x                   
            for testCenter in clusters[clusNum][1:]:
                localDSum=0  #set testCenters cost to 0
                for testPoint in clusters[clusNum]:  #iterate through x's in that cluster                  
                    localDSum+=np.linalg.norm(testPoint-testCenter)**2  #add costs to local sum
                if localDSum<minDSum:   #if local sum less than previous lowest
                    minDSum=localDSum   #update min to be this sum
                    centerP=X[testCenter,:]    #update center of this cluster (at index clusNum) to be the centerPoint
                    medoidsIndexList[clusNum] = testCenter
            init[clusNum,:]= centerP        #assign this clusters index row in the Z matrix to be the centerP in the Xmatri
            
            #3. caluclate cost based on reassigned x's and avg z's
            newTotalAvgCost+= minDSum/len(clusters[clusNum])  #add the average of this cost to new total
        cost[1]=newTotalAvgCost
           
    clusterAssignments= [0 for x in range(numXRows)]
    for clusterNum in range(len(clusters)):
         for point in clusters[clusterNum] :
             clusterAssignments[point]=clusterNum 
    for clus in range(len(clusters)):
        graphAMatrix(X,clusters[clus],clus)
    graphCenters(init)
    
    return(init,medoidsIndexList,np.array(clusterAssignments))
     
##average a certain person number's faces, return plot        
def averageSomeStuffs(personNum):
    D=getData()
    y=D[1]
    X=D[0]
    X1, y1 = limitPics(X,y,[personNum],100)
    some= 0
    for image in X1:
        some+=image
    avg=some/100
    showIm(avg)
#averages 
def myInitAverager(X,y,k):
    XGeorge= 0
    XBarb=0
    SumG=0
    SumB=0
    (n, d) = X.shape
    init = np.zeros((k, d), dtype=float)
    for xrow in range(40):
        SumG+= X[xrow,:]
    XGeorge=SumG/40
    for x2row in range(40,80):
        SumB+=X[x2row,:]
    XBarb=SumB/40
    init[0,:]=XGeorge
    init[1,:]=XBarb
    return init
#does PCA analysis up to k principal components   
def reduceDimensions(k,E):
    X=getData()[0]
    sub= E[:,0:k]
    Z= np.dot(X,sub)
    Xp= np.dot(Z,sub.T)
    plotGallery([vecToImage(Xp[i]) for i in range(12)])
    
def reduceDGetScore(X1,y1,pcs):
    E,mu=PCA(X1)
    sub= E[:,0:pcs]
    Z= np.dot(X1, sub)
    Xp=np.dot(Z,sub.T)
    newInit= cheatInit(Xp, y1,2)
    newClusters= kmedoids(Xp, newInit)
    return scoreMedoids(newClusters, y1)
    
def plotPCs(X1, y1):
    xyList=[[],[]]
    for pcn in range(1,40,10):
        xyList[0].append(pcn)
        xyList[1].append(reduceDGetScore(X1,y1,pcn))    
        plt.plot(pcn,reduceDGetScore(X1, y1, pcn))
    plt.show()
    return xyList
    
# Provides an initial set of data points to use to initialize
# clustering.  picks k random unique data points for medoids.
# Input:
# - X: n by d array representing n d-dimensional data points
# - y: 1 by n array representing integer class labels
# - k: number of classes
# Output:
# - init: k by d array representing initial cluster medoids
def myInit(X, y, k):
    initList= random.sample(range(0, X.shape[0]),k)
    initArray= np.zeros((k,X.shape[1]))
    for i in range(k):
        initArray[i,:]=np.array(X[initList[i],:])
    return initArray
    

#dimensions for pca
lDimensions= [1, 10, 50, 100, 500, 1288]
#my little array for problems 7 and testing purposes
myX=np.array([[-3,3],[-5,3],[-3,5],[-5,5],[3,-3],[3,-5],[5,-5],[-3,-3],[-5,-3],[-3,-5],[5,-3],[-5,-5]])
myI= np.array([[ 3., -3.], [-3., -3.],[-3.,  3.]])

#cen, clus, cenin= kmedoids(myX, myI)
data= getData()
X= data[0]
y= data[1]
#X1, y1 = limitPics(X,y, [4,13],40)
#cheatI= cheatInit(X1, y1, 2)
#centroids,cenIndices,clusterAssignments = kmedoids(X1, cheatI)
#clustering = kmedoids(X1, cheatI)
#myI= myInit(X1,y1,2)
#myclustering= kmedoids(X1, myI)
#scoreMedoids(myclustering,y1)

def scoreMyInit(X1,y1,k,runs):
    score=[]
    for i in range(runs):
        newInit= myInit(X1, y1,k)
        newClusters= kmedoids(X1, newInit)
        score.append(scoreMedoids(newClusters, y1))
    return score
    
def scoreCheatInit(X1,y1,k,runs):
    score=[]
    for i in range(runs):
        newInit= cheatInit(X1, y1,k)
        newClusters= kmedoids(X1, newInit)
        score.append(scoreMedoids(newClusters, y1))
    return score

def findRel(X,y,p1,p2):
    Xo,yo = limitPics(X,y,[p1,p2],40)
    return scoreCheatInit(Xo,yo,2,1)
    
def PCAOutputMatrix(X,k,E):
    sub= E[:,0:k]
    Z= np.dot(X,sub)
    Xp= np.dot(Z,sub.T)
    return Xp
    
def CvsScore(X,y,E):
    Xp= PCAOutputMatrix(X,100,E)
    newY = [+1 if yi == 4 else -1 for yi in y] 
    (X1, X2, y1, y2) = sklearn.cross_validation.train_test_split(Xp,newY, test_size=.75)   
    Clist= [.001, .01 , .1, 1.0 ,10.0]
    scores=[]
    trainScores=[]
    for Cval in Clist:
        clf = SVC(kernel='linear', C=Cval, gamma=0) 
        clf.fit(X1,y1) 
        scores.append(clf.score(X2,y2))
        trainScores.append(clf.score(X1,y1))  
    
    return [Clist, scores, trainScores]
    
def CvsScoreC100(X,y,E):
    l = [x for x in range(0,300,99)]
    scores=[]
    trainScores=[]
    for i in range(0,300,20):
        Xp= PCAOutputMatrix(X,i,E)
        newY = [+1 if yi == 4 else -1 for yi in y] 
        (X1, X2, y1, y2) = sklearn.cross_validation.train_test_split(Xp,newY, test_size=.75)         
        clf = SVC(kernel='linear', C=100, gamma=0)  
        clf.fit(X1,y1)        
        scores.append(clf.score(X2,y2))
        trainScores.append(clf.score(X1,y1))

        
    return [l, scores, trainScores]

    
                  
