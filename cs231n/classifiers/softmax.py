import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
    
  loss = 0.0
  num_training = X.shape[0]
  num_classes = W.shape[1]

  # Computing the softmax loss
  scores = np.dot(X, W)
  exp_scores = np.exp(scores)
  prob = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
  loss_arr = np.zeros([num_training, ])
  for i in xrange(num_training):
        loss_arr[i] = -np.log(prob[i][y[i]])
  
  data_loss = np.sum(loss_arr)/num_training
  reg_loss = 0.5*reg*np.sum(np.dot(W, W.T))
  loss = data_loss + reg_loss

  # Updating parameters using gradient descent
  dW = np.zeros_like(W)
  dscores = np.zeros_like(prob)
  for i in xrange(num_training):
    for j in xrange(num_classes):
        if j == y[i]:
            dscores[i][j] = prob[i][j] - 1
        else:
            dscores[i][j] = prob[i][j]
            
  dscores /= num_training
  dW = np.dot(X.T, dscores)
  dW += reg*W
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_training = X.shape[0]
  num_classes = W.shape[1]
  scores = np.dot(X, W)
  exp_scores = np.exp(scores)
  probs = exp_scores/ np.sum(exp_scores, axis=1, keepdims=True)
  loss_arr = -np.log(probs[range(num_training),y])
  data_loss = np.sum(loss_arr)/num_training
  reg_loss = np.sum(0.5*reg*(W*W))
  loss = data_loss + reg_loss
  dscores = probs
  dscores[range(num_training), y] -= 1
  dscores /= num_training
  dW = np.dot(X.T, dscores)
  dW += reg*W
    

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
#   pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

