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
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_data = X.shape[0]
  num_class = W.shape[1]
  for i in np.arange(num_data):
    scores = X[i].dot(W)
    loss += -scores[y[i]] + np.log(np.sum(np.exp(scores)))
    for j in np.arange(num_class):
      if j == y[i]:
        dW[:, j] += (-1 + np.exp(scores[j]) / np.sum(np.exp(scores))) * X[i]
      else:
        dW[:, j] += (np.exp(scores[j]) / np.sum(np.exp(scores))) * X[i]
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  loss /= num_data
  dW /= num_data

  loss += reg * np.sum(W * W)
  dW += reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_data = X.shape[0]
  num_class = W.shape[1]
  scores = X.dot(W)
  loss = np.sum(np.log(np.sum(np.exp(scores), axis = 1))) - np.sum(scores[np.arange(num_data), y])

  scores_expsum = np.sum(np.exp(scores), axis = 1)
  coefficient = np.zeros((num_data, num_class))
  coefficient += np.exp(scores) / scores_expsum.reshape(-1, 1)
  coefficient[np.arange(num_data), y] -= 1
  dW = X.T.dot(coefficient)

  loss /= num_data
  loss += reg * np.sum(W * W)
  dW /= num_data
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

