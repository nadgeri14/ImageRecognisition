import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):


  def __init__(self, input_size, hidden_size, output_size, std=1e-2):
  
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
  
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
  
  
    y1 = X.dot(W1) + b1 #(N,H) + (H)
    h1 = np.maximum(0, y1)
    y2 = h1.dot(W2) + b2
    scores = y2
  
    
    if y is None:
      return scores

    # Compute the loss
    loss = None

    exp_scores = np.exp(scores)
    probality = exp_scores/(np.sum(exp_scores,axis =1, keepdims = True))
    correct_logprobs = -np.log(probality[range(N),y])
    loss = np.sum(correct_logprobs)/N
    reg_loss = 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
    loss += reg_loss
 
    # Backward pass: compute gradients
    grads = {}

    dy2 = probality
    dy2[range(N),y] -= 1
    dy2 /= N 
    dW2 = h1.T.dot(dy2)
    dh1 = dy2.dot(W2.T)
    dy1 = dh1 * (y1 >= 0)
    dW1 = X.T.dot(dy1)
    dW1 += reg * W1
    dW2 += reg * W2
    db1 = np.sum(dy1, axis=0)
    db2 = np.sum(dy2, axis=0)

    grads['W1'] = dW1
    grads['W2'] = dW2
    grads['b1'] = db1
    grads['b2'] = db2


    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):

    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []
    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      indices = np.random.choice(num_train,batch_size,replace = True)
      X_batch = X[indices]
      y_batch = y[indices]


      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)


      self.params['W1'] += -learning_rate * grads['W1']
      self.params['W2'] += -learning_rate * grads['W2']
      self.params['b1'] += -learning_rate * grads['b1']
      self.params['b2'] += -learning_rate * grads['b2']


      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):

    y_pred = None


    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    h1 = np.maximum(0,(X.dot(W1)+b1))
    y2 = h1.dot(W2) + b2
    y_pred = np.argmax(y2, axis=1)


    return y_pred


