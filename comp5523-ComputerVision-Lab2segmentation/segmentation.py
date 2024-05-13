"""
COMP5523 - Computer Vision and Image Processing
Lab 2
Author: XXXXX
Python Version: 3.5+
note 1: this code is  mainly adapted from https://github.com/StanfordVL/CS131_release
note 2: You may find various solutions by others in Google search. But it merely dulls your brain.
note 3: I will compare your codes with these solutions and other students` solutions by professional software.
Please show me your own ideas by codes!
"""

import numpy as np
import random
from scipy.spatial.distance import squareform, pdist
from skimage.util import img_as_float
from numpy import clip

### Clustering Methods
def kmeans(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2
    Hints
    - np.allclose may help or you can try other stop tricks

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape
    #print('Feature Shape: Row', N , " Column:", D)
    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False) # initial random select point 
    centers = features[idxs]
    #print("initial center: ", centers)
    assignments = np.zeros(N, dtype=np.uint32)
    #print("assignments shape :", assignments.shape)
    old_assignments = np.zeros(N, dtype=np.uint32)  # 
    
    for n in range(num_iters):
        ### YOUR CODE HERE
        # this code base on loop and iteration the number of update new center
        for numPoint in range(N):
            distance = {}
            for i in range(k): 
                distance[i] =np.sqrt((features[numPoint][0]- centers[i][0])**2 + 
                          (features[numPoint][1]- centers[i][1])**2 ) # calculate distance from each point to centre 
                #print("Num Point ", numPoint, ": Distance for center ", i, " : ", distance[i])
            #find the closest cluster
            cluster = min(distance, key=distance.get) 
            assignments[numPoint] = cluster 
            #print('Num Point ', numPoint, ' Assign Cluster :', cluster)
        #update the new centre after assign all point 
        for j in range(k):
           # print("old center ", j , " : ", centers[j])
            assignedPoint = [features[pos] for pos in range(N) if assignments[pos] == j] #collect each assignedPoint for specific center
            centers[j]= np.mean(assignedPoint, axis=0) # calculate new center 
           # print("New centre ", j, " : ", centers[j] )
        # check cluster assignments did not change
        if np.allclose(old_assignments, assignments):
            #print("The cluster assignments did not change , should be stop the iteration.")
            print("Num of Iters: ", n)
            break
        else:
            #update the old_assignment
            old_assignments = np.copy(assignments)
        ### END YOUR CODE

    return assignments

def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find np.repeat and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)
    old_assignments = np.zeros(N, dtype=np.uint32)  # 

    for n in range(num_iters):
        ### YOUR CODE HERE
        #new faster algorithm  implement base on matrix operation 
        # Compute the N*k matrix that contains squared Euclidean distances
	    # using the fast algorithm based on BLAS-3 operations given in class
	    # feature: N*D matrix
	    # centers: k*D matrix
     	# new Dimension: N*k matrix (should rewrite this matrix)
        normedFeatures = np.linalg.norm(features, axis= 1)
        #print("Norm Feature shape ",normedFeatures.shape, "  :", normedFeatures[:10])
        normedFeatures= np.power(normedFeatures, 2).reshape(N,1)
        #print("Norm Feature shape ",normedFeatures.shape, "  :", normedFeatures[:10])
        centerDotProduct = 2.0 * features.dot(centers.T)  # dot product  feature and center 
        #print('Center Dot Product',centerDotProduct[:10] )
        normedCenter = np.linalg.norm(centers, axis=1)
        normedCenter  = np.power(normedCenter, 2)
        #print('normed Center ',centerDotProduct[:10] )
        distance = normedFeatures- centerDotProduct +normedCenter
        #print("distance shape ", distance.shape, " : ", distance[:10] )
        assignments = np.nanargmin(distance, axis=1)  # assign the clustering by min distance 
        #update the new centre after assign all point 
        for j in range(k):
            #print("old center ", j , " : ", centers[j])
            assignedPoint = [features[pos] for pos in range(N) if assignments[pos] == j] #collect each assignedPoint for specific center
            centers[j]= np.mean(assignedPoint, axis=0) # calculate new center 
            #print("New centre ", j, " : ", centers[j] )
        # check cluster assignments did not change
        if np.allclose(old_assignments, assignments):
            #print("The cluster assignments did not change , should be stop the iteration.")
            print("Num of Iters: ", n)
            break
        else:
            #update the old_assignment
            old_assignments = np.copy(assignments)
        ### END YOUR CODE

    return assignments



### Pixel-Level Features
def color_features(img):
    """ Represents a pixel by its color.

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    H, W, C = img.shape
    print('Image dimension : Height ', H, ' Width ', W , " Channel ", C)
    img = img_as_float(img)
    #print("img new shape :", img.shape)
    features = np.zeros((H*W, C))
    ### YOUR CODE HERE
    #print('Color Feature shape ', features.shape)
    
    #features = np.copy(img)
    features = img.reshape(H*W, C) # reshape the image dimension from (H,W C) to (H*W, C)
    return features

def color_position_features(img):
    """ Represents a pixel by its color and position.

    Combine pixel's RGB value and xy coordinates into a feature vector.
    i.e. for a pixel of color (r, g, b) located at position (x, y) in the
    image. its feature vector would be (r, g, b, x, y).

    Don't forget to normalize features.

    Hints
    - You may find np.mgrid and np.dstack useful
    - You may use np.mean and np.std

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C+2)
    """
    H, W, C = img.shape
    print('Image dimension : Height ', H, ' Width ', W , " Channel ", C)
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))
    
    ### YOUR CODE HERE
    #print('Image dimension : Height ', H, ' Width ', W , " Channel ", C)
    #print('image raw: ', img[:5])
    #fill the image and x y p
    for y in range(H):
        for x in range(W):
            if C== 3:
                features[x+y * W, 0]= img[y][x][0] # fill r channel data
                features[x+y * W, 1]= img[y][x][1]
                features[x+y * W, 2]= img[y][x][2]
                features[x+y * W, 3]= x
                features[x+y * W, 4]= y
            
            if C == 2:
                features[x+y * W, 0]= img[y][x][0] # fill r channel data
                features[x+y * W, 1]= img[y][x][1]
                features[x+y * W, 2]= x
                features[x+y * W, 3]= y
    #print("Feature shape ", features.shape, "After with x, y: ", features[:5])
    features = np.array(features).astype(float) # convert to float 
    #standization 1 (Gloabl standardization)
    # mean = features.mean()
    # std =  features.std()
    # features = (features -mean)/ std
    # standization 2 positive  (Gloabl standardization)
    mean = features.mean()
    std =  features.std()
    features = (features -mean)/ std
    # clip pixel values to [-1,1]
    features = clip(features, -1.0, 1.0)
    # shift from [-1,1] to [0,1] with 0.5 mean
    features = (features + 1.0) / 2.0
    #print("Feature shape ", features.shape, "After norm with x, y: ", features[:5])
    ### END YOUR CODE
    
    return features

def my_features(img):
    """ Implement your own features

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    features = None
    H, W, C = img.shape
    print('Image dimension : Height ', H, ' Width ', W , " Channel ", C)
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))
    ### YOUR CODE HERE
    #print('Image dimension befor float : Height ', H, ' Width ', W , " Channel ", C)
    #fill the image and x y p
    for y in range(H):
        for x in range(W):
            if C== 3:
                features[x+y * W, 0]= img[y][x][0] # fill r channel data
                features[x+y * W, 1]= img[y][x][1]
                features[x+y * W, 2]= img[y][x][2]
                features[x+y * W, 3]= x
                features[x+y * W, 4]= y
            if C == 2:
                features[x+y * W, 0]= img[y][x][0] # fill r channel data
                features[x+y * W, 1]= img[y][x][1]
                features[x+y * W, 2]= x
                features[x+y * W, 3]= y
    #print("Feature shape ", features.shape, "After with x, y: ", features[:5])
    features = np.array(features).astype(float) # convert to float 
    # normalize the matrix  ()
    # matMin = np.min(features)
    # matMax = np.max(features)
    # for col in range(W):
    #     for row in range(H):
    #         #base on nomralization (value - min)/ (max-min)  , range 0 to 1
    #         #print("mean :", matMin, " Max : ", matMax)
    #         if C== 3:
    #             features[col+row * W, 0]= (features[col+row * W, 0] - matMin)/ ( matMax- matMin) # fill r channel data
    #             features[col+row * W, 1]= (features[col+row * W, 1] - matMin)/ ( matMax- matMin) 
    #             features[col+row * W, 2]= (features[col+row * W, 2] - matMin)/ ( matMax- matMin) 
    #             features[col+row * W, 3]= (features[col+row * W, 3] - matMin)/ ( matMax- matMin) 
    #             features[col+row* W, 4]= (features[col+row * W, 4] - matMin)/ ( matMax- matMin) 
    #         if C == 2:
    #             features[col+row * W, 0]= (features[col+row * W, 0] - matMin)/ ( matMax- matMin) # fill r channel data
    #             features[col+row * W, 1]= (features[col+row * W, 1] - matMin)/ ( matMax- matMin) 
    #             features[col+row * W, 2]= (features[col+row * W, 2] - matMin)/ ( matMax- matMin) 
    #             features[col+row * W, 3]= (features[col+row * W, 3] - matMin)/ ( matMax- matMin) 
    
    #standization 2 (local standardization) per channel
    mean = features.mean(axis =(0,1))
    std =  features.std(axis =(0,1))
    features = (features -mean)/ std
    ### END YOUR CODE
    return features


### Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    accuracy = None
    ### YOUR CODE HERE
    H, W = mask_gt.shape    
    # print("Mask gt shape:", mask_gt.shape)
    # print("mask shape: ", mask.shape )
    tp = 0  # true positive
    tn = 0  # true negative
    fn = 0  # false negative
    fp = 0  # false positive
    
    
    # collect tp 
    for i in range(H):
        for j in range(W):
            if  mask_gt[i][j] == 1 and  mask[i][j] == 1:
                tp +=1  # true 
            if  mask_gt[i][j] == 0 and  mask[i][j] == 0:
                tn +=1  # true neg
            if  mask_gt[i][j] == 1 and  mask[i][j] == 0:
                fn +=1
            if  mask_gt[i][j] == 0 and  mask[i][j] == 1:
                fp +=1
                
    accuracy = float((tp+tn) / (tp+tn + fn + fp))
    
    ### END YOUR CODE

    return accuracy

def evaluate_segmentation(mask_gt, segments):
    """ Compare the estimated segmentation with the ground truth.

    Note that 'mask_gt' is a binary mask, while 'segments' contain k segments.
    This function compares each segment in 'segments' with the ground truth and
    outputs the accuracy of the best segment.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        segments - An array of the same size as mask_gt. The value of a pixel
            indicates the segment it belongs.

    Returns:
        best_accuracy - Accuracy of the best performing segment.
            0 <= accuracy <= 1, where 1.0 indicates a perfect segmentation.
    """

    num_segments = np.max(segments) + 1
    best_accuracy = 0

    # Compare each segment in 'segments' with the ground truth
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy
