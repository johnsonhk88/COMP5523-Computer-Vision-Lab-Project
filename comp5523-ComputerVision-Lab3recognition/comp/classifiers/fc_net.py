from builtins import range
from builtins import object
import numpy as np

from comp.layers import *
from comp.layer_utils import *


'''
Name: Chong Kit Sang 
Student ID: 19005168g

'''


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.
    HINT: use the affine_relu_forward(*) module in the layers.py

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        Note that you can change these default values according to your computer.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.cache = {}  # define cache 
        
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)  # inital weight 1
        self.params['b1'] = np.zeros(hidden_dim)  #inital with zero at bias 1
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes) # inital weight 2
        self.params['b2'] = np.zeros(num_classes)
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None

        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # get W1, b1 and W2 , b2 paramter
        W1 = self.params['W1'] 
        b1 = self.params['b1']
        W2 = self.params['W2']  
        b2 = self.params['b2']
        
        hidden, self.cache['hidden']  = affine_relu_forward(X, W1, b1)  # input for affine relue 
        scores,  self.cache['output']  = affine_forward(hidden, W2, b2)  # input for affline forward
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        # google it if you do not what is L2)                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # sofmax loss
        loss , delta1 = softmax_loss(scores, y) 
        
        # apply regularization 
        loss = loss + 0.5 * self.reg * np.sum(W1**2) + 0.5 * self.reg * np.sum(W2**2)
        
        #backpropagation
        delta2 , grads['W2'], grads['b2'] = affine_backward(delta1, self.cache['output'])
        _, grads['W1'], grads['b1'] = affine_relu_backward(delta2, self.cache['hidden'])
        
        #sum of gradient for regularization
        grads['W2']  += self.reg *W2
        grads['W1']  += self.reg *W1
        
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also include
    dropout and batch/layer normalization as options. For a network with L layers,
    the bonus architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

     where batch/layer normalization and dropout are provided to you for free, and the {...} block is
    repeated L - 1 times.

    Note: you can use the affine_norm_relu_forward and affine_norm_relu_backard in layer_util.

    On the other hand, if you do not know how to use [batch/layer norm] and dropout, just use the architecture of
        {affine-relu} X (L - 1) - affine - softmax
    But you will not receive the bonus pts.


    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.


    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=1, normalization=None, reg=0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        self.cache = {}

        ############################################################################
        # TODO: Initialize the parameters of affine layers of the network, storing #
        # all values in the self.params dictionary.                                #
        # Store weights and biases for the first layer                             #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        # Note you only require initialize the affine layer params.                #
        ############################################################################
        #  it is for basic verion 
      
        
        
        #pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        ############################################################################
        # TODO: Initialize the parameters of batch/layer norm of the network, storing #
        # all values in in the self.params dictionary                              #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        # Remark: this part is optional. Leave it empty if you do not want the Bonus #
        ############################################################################
        # it is for advance version
          
        for layer in range(self.num_layers):
            # initial weigth and biase (input and hidden layer)
            if layer ==0:
                self.params['W'+ str(layer+1)] = weight_scale * np.random.randn(input_dim, hidden_dims[layer])
                self.params['b'+ str(layer+1)] = np.zeros(hidden_dims[layer])
                # gamma and beta  for normalization
                if self.normalization is not None:
                    self.params['gamma'+ str(layer+1)] = np.ones(hidden_dims[layer])
                    self.params['beta'+ str(layer+1)] = np.zeros(hidden_dims[layer])
            #initial weight and biase (hidden to hidden layer)        
            elif layer < self.num_layers -1 :
                self.params['W'+ str(layer+1)] = weight_scale * np.random.randn(hidden_dims[layer-1], hidden_dims[layer])
                self.params['b'+ str(layer+1)] = np.zeros(hidden_dims[layer])
                # gamma and beta  for normalization
                if self.normalization is not None:
                    self.params['gamma'+ str(layer+1)] = np.ones(hidden_dims[layer])
                    self.params['beta'+ str(layer+1)] = np.zeros(hidden_dims[layer])
            #inital weight and biase (hidden to output)
            else:
                self.params['W'+ str(layer+1)] = weight_scale * np.random.randn(hidden_dims[layer-1], num_classes)
                self.params['b'+ str(layer+1)] = np.zeros(num_classes)
                
        
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        # Note: switch this block off if the Bonus is not your concern.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        # Note: switch this block off if the Bonus is not your concern.
        self.bn_params = []
        if self.normalization == 'batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization == 'layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization == 'batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        cache = {}
        #forward pass 
        layer_input = {'layer0': X}
        for layer in range(self.num_layers):
            #load parameter
            W, b = self.params['W'+str(layer+1)], self.params['b'+ str(layer+1)]
            l, l_prev = 'layer'+ str(layer+1) , 'layer'+ str(layer)
            
            if mode == 'train':
                bn_param = {'mode': 'train'}
            else:
                bn_param = {'mode': 'test'}
            
            #pass to hidden layer
                
            if layer < self.num_layers -1 :
                if self.normalization is not None:
                    gamma , beta = self.params['gamma'+ str(layer+1)] , self.params['beta'+ str(layer+1)]
                    layer_input[l], self.cache[l] = affine_norm_relu_forward(layer_input[l_prev], W, b, gamma, beta, 
                                                                          bn_param, self.normalization,  self.use_dropout , self.dropout_param)
                else: 
                    layer_input[l], self.cache[l] = affine_relu_forward(layer_input[l_prev], W, b)
            
            # hidden to output layer
            else:
                layer_input[l], self.cache[l] = affine_forward(layer_input[l_prev], W, b)
                
        
        
        scores = layer_input['layer'+ str(self.num_layers)]
        
       # pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        #last layer 
        last_layer = self.num_layers
        
        #error at eah layer
        d = {}
        
        #computing softmax loss 
        loss , dout = softmax_loss(scores, y)
        grads = {}
        
        # backpropagation from output layer
        w = 'W' + str(last_layer)
        b = 'b' + str(last_layer)
        c = 'layer' + str(last_layer)
        
        dh, grads[w], grads[b] = affine_backward(dout, self.cache[c])
        loss += 0.5* self.reg * np.sum(self.params[w]**2)
        grads[w] += self.reg * self.params[w]
        
        #backpropagation from hidden layer
        for layer in reversed(range(last_layer -1)):
             w = 'W' + str(layer+1)
             b = 'b' + str(layer+1)
             gamma = 'gamma' + str(layer+1)
             beta = 'beta' + str(layer+1)
             c = 'layer' + str(layer+1)
             
             if self.normalization is not None:
                 dh, grads[w], grads[b], grads[gamma], grads[beta] =affine_norm_relu_backward(dh, self.cache[c], self.normalization, self.use_dropout)
             else:
                 dh, grads[w], grads[b] = affine_relu_backward(dh, self.cache[c])
            
             loss += 0.5 * self.reg * np.sum(self.params[w]**2) # cost for regularization term
             grads[w] += self.reg * self.params[w] # gradient for regularization term
             
                 
        
        
        #pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads



