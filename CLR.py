from keras.optimizers import *

if K.backend()=='tensorflow':
    import tensorflow as tensor
else:
    import theano.tensor as tensor
    
class SGD_CLR(Optimizer):
    """Basic Stochastic gradient descent optimizer.
    Supports triangular cyclic learning rate
    Includes attribute self.current_lr for testing purposes
    """

    def __init__(self, lr=0.01, max_lr=0.06, step_size=2000., **kwargs):
        super(SGD_CLR, self).__init__(**kwargs)
        self.iterations = K.variable(0., name='iterations')
        self.lr = K.variable(lr, name='lr')
        self.step_size = K.variable(step_size, name='step_size')
        self.max_lr = K.variable(max_lr, name='max_lr')
        self.current_lr = K.variable(0., name='current_lr')

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        cycle = tensor.floor(1+self.iterations/(2*self.step_size))
        x = K.abs(self.iterations/self.step_size - 2*cycle + 1)
        lr = self.lr + (self.max_lr-self.lr)*K.maximum(0., (1-x))
        self.updates.append(K.update(self.current_lr, lr))
        # momentum
        shapes = [K.get_variable_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g in zip(params, grads):
            new_p = p - lr * g
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'iterations': float(K.get_value(self.iterations)),
                  'step_size': float(K.get_value(self.step_size)),
                  'max_lr': float(K.get_value(self.max_lr)),
                  'current_lr': float(K.get_value(self.current_lr))
                 }
        base_config = super(SGD_CLR, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))