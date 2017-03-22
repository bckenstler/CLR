# CLR
This repository includes modified Keras optimizers that allow implementation of three cyclical learning rate policies, as detailed in this paper https://arxiv.org/abs/1506.01186.

clr_optimizers.py is a modified version of Keras' default optimizer module (https://github.com/fchollet/keras/blob/master/keras/optimizers.py) that includes 3 CLR policies, '''triangular''', '''triangular2''', and '''exp_range'''.

Each optimizer can now implement cyclical learning rate with one of the following three approaches:

"triangular":

This method is a simple triangular cycle.

To use triangular clr, simply pass into any optimizer the following:
clr = {
    "mode":"triangular",
    "step_size":step_size,
    "max_lr":max_lr
    }
    
Where step_size is half the period of the cycle in iterations,
and max_lr is the peak of the cycle (default lr is the base lr).

"triangular2":

This method is a triangular cycle that decreases the cycle amplitude by half after each period, while keeping the base lr constant.

To use triangular2 clr, simply pass into any optimizer the following:
clr = {
    "mode":"triangular2",
    "step_size":step_size,
    "max_lr":max_lr
    }
    
Where step_size is half the period of the cycle in iterations,
and max_lr is the peak of the cycle (default lr is the base lr).

"exp_range":

This method is a triangular cycle that scales the cycle amplitude by a factor `gamma**(iterations)` at each iteration.
In other words, the base lr remains the same, but the range between between base lr and peak lr is reduced by half each cycle.
To use triangular clr, simply pass into any optimizer the following:
clr = {
    "mode":"exp_range",
    "step_size":step_size,
    "max_lr":max_lr,
    "gamma": gamma
    }
    
Where step_size is half the period of the cycle in iterations,
and max_lr is the peak of the cycle (default lr is the base lr).

Note that clr happens prior to any further learning rate adjustments as called for in a given optimizer, with the exception of learning rate decay. If implemented, decay happens prior to clr; this alters the cycle by lowering the minimum lr; the max_lr is kept constant.

clr_optimizer_tests.ipynb contains tests demonstrating desired behavior of optimizers in comparison to toy functions.
