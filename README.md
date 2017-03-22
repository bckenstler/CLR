# CLR
This repository includes a simple implementation of a triangular cyclical learning rate approach, as detailed here https://arxiv.org/abs/1506.01186 .

clr_optimizers.py is a modified version of Keras' default optimizer module (https://github.com/fchollet/keras/blob/master/keras/optimizers.py) that includes triangular CLR.

Each optimizer can now implement triangular cyclical learning rate (clr).
To use clr, simply pass into any optimizer the following:
clr = {
    "step_size":step_size,
    "max_lr":max_lr
    }
Where step_size is half the period of the cycle in iterations,
and max_lr is the peak of the cycle (default lr is the base lr).

clr_optimizer_tests.ipynb contains tests demonstrating desired behavior.
