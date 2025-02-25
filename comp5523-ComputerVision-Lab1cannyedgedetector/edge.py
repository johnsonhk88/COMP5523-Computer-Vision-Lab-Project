"""
COMP5523 - Computer Vision and Image Processing
Lab 1 Homework
Author: Chong Kit Sang
Studend ID: 19005168g
Python Version: 3.5+
note 1: this code is  mainly adapted from https://github.com/StanfordVL/CS131_release
note 2: You may find various solutions by others in Google search. But it merely dulls your brain.
note 3: I will compare your codes with these solutions and other students` solutions by professional software.
Please show me your own ideas by codes!
"""

import numpy as np

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape  # get image heigth and width
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))  # set all output zero

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    ### implement convolution filp x, filp y (filp verital, and then file horizontal)
    kernel = np.flip(np.flip(kernel, 0), 1)
    
    for y in range(Hi):
        for x in range(Wi):
            out[y, x] = np.sum(padded[y:y+ Hk, x:x+Wk] * kernel)
    print('convolution result:\n\r',out)        
    ### END YOUR CODE
    
    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """

    kernel = np.zeros((size, size))

    ### YOUR CODE HERE
    for i in range(size):
        for j in range(size):
            # calucate each matrix element from gaussian function
            kernel[i, j] = (1/(2 * np.pi *sigma**2)) *  \
                np.exp( -((i-size//2 )**2 +(j-size//2)**2) / float(2* sigma**2))
    print('Gaussian Kernel:\n\r',kernel) # for debug using
    ### END YOUR CODE
    
    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    filterkernel = 1. / 2 * np.array([[1, 0, -1]])
    print("filterkernel shape: ", filterkernel.shape ,"\n\rfilterkernel:\n\r", filterkernel )
    out = conv(img, filterkernel)
    ### END YOUR CODE

    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    filterkernel = 1./ 2 * np.array([[1], [0], [-1]])      # filter 1, 0, -1 for y-axis
    print("filterkernel shape: ", filterkernel.shape ,"\n\rfilterkenel:\n\r", filterkernel )
    out = conv(img, filterkernel)
    ### END YOUR CODE

    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    # Compute partial derivatives of smoothed image
    Gx = partial_x(img)
    Gy = partial_y(img)
    #Calulate the magitude of graident  
    G = np.sqrt(Gx**2 + Gy**2)
    theta =  (np.rad2deg(np.arctan2(Gy, Gx))+180)%360
    print('magitude:\n\r', G)
    print('Theta:\n\r', theta)
    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45

    ### BEGIN YOUR CODE
    for i in range(1, H-1):
        for j in range(1, W-1):
            rad = np.deg2rad(theta[i, j]) # convert to rad 
            #find close 8 direction value 
            th1 = G[i-int(np.round(np.sin(rad))), j - int(np.round(np.cos(rad)))] # r
            th2 = G[i+int(np.round(np.sin(rad))), j + int(np.round(np.cos(rad)))]
            #suppressed non-maximum value to zero
            if not (G[i, j]>= th1 and G[i, j] >= th2):
                out[i, j] =0
            else:
                out[i, j] = G[i,j] # set maximum value
    print('Non-maximum out:\n\r' ,out)             
    ### END YOUR CODE

    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = np.zeros(img.shape, dtype=np.bool)
    weak_edges = np.zeros(img.shape, dtype=np.bool)

    ### YOUR CODE HERE
    #classify strong and weak edges
    strong_edges = img > high  
    weak_edges = (img > low) & (img< high)
    ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).
    
    Returns:
        edges: numpy boolean array of shape(H, W).
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=np.bool)

    # Make new instances of arguments to leave the original
    # references intact
    weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)

    ### YOUR CODE HERE
    for i in range(1, H-1):
        for j in range(1, W-1):
            neighbor = get_neighbors(j, i, H, W)
            if weak_edges[i, j] and np.any(edges[x,y] for x, y in neighbor):
                edges[i, j] = True  # set true 
    ### END YOUR CODE

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    ### YOUR CODE HERE
    #put all process together 
    kernal = gaussian_kernel(kernel_size, sigma)  #caluate gaussian kernal
    smoothedImg = conv(img, kernal)  # Generate smoothed img by convolution
    G, theta = gradient(smoothedImg) # find gradient 
    nonMaxSuppressImg = non_maximum_suppression(G, theta)  # apply non-maximum suppression
    #appy double threshold to find strong and weak edge
    strong_edges, weak_edges = double_thresholding(nonMaxSuppressImg, high, low)  
    edge = link_edges(strong_edges, weak_edges) # link strong and weak eadge to find real edge
    ### END YOUR CODE

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W).
        
    Returns:
        accumulator: numpy array of shape (m, n).
        rhos: numpy array of shape (m, ).
        thetas: numpy array of shape (n, ).
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    '''
    Extract edges of the image using Canny
    1 Initialize parameter space rs , thetas
    2 Create accumulator array and initialize to zero
    3 for each edge pixel
    4 for each theta
    5 Calculate r = x cos(theta) + y sin(theta)
    6 Increment accumulator at r, theta
    7 Find maximum values in accumulator (lines)
     Extract related r, theta Algorithm
    '''
    ### YOUR CODE HERE
    for i, j in zip(ys, xs):
        for angleIndex in range(thetas.shape[0]): #
            polor = j * cos_t[angleIndex] + i * sin_t[angleIndex] # transform to polor 
            accumulator[int(polor + diag_len), angleIndex] +=1  #add 1
    ### END YOUR CODE

    return accumulator, rhos, thetas
