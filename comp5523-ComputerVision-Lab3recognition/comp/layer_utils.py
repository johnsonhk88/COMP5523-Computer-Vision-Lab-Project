pass
from comp.layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

def affine_norm_relu_forward(x, w, b, gamma, beta, bn_param, normalization, dropout, do_param):
    bn_cache, do_cache = None, None
    # affine layer
    out, fc_cache = affine_forward(x, w, b)
    # batch/layer norm
    if normalization == 'batchnorm':
        out, bn_cache = batchnorm_forward(out, gamma, beta, bn_param)
    elif normalization == 'layernorm':
        out, bn_cache = layernorm_forward(out, gamma, beta, bn_param)
    # relu
    out, relu_cache = relu_forward(out)
    # dropout
    if dropout:
        out, do_cache = dropout_forward(out, do_param)
    return out, (fc_cache, bn_cache, relu_cache, do_cache)


def affine_norm_relu_backward(dout, cache, normalization, dropout):
    fc_cache, bn_cache, relu_cache, do_cache = cache
    # dropout
    if dropout:
        dout = dropout_backward(dout, do_cache)
    # relu
    dout = relu_backward(dout, relu_cache)
    # batch/layer norm
    dgamma, dbeta = None, None
    if normalization == 'batchnorm':
        dout, dgamma, dbeta = batchnorm_backward_alt(dout, bn_cache)
    elif normalization == 'layernorm':
        dout, dgamma, dbeta = layernorm_backward(dout, bn_cache)
    # affine layer
    dx, dw, db = affine_backward(dout, fc_cache)
    return dx, dw, db, dgamma, dbeta

