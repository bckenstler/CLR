# CLR
This repository includes a simple implementation of a triangular cyclical learning rate approach, as detailed here https://arxiv.org/abs/1506.01186 .

clr_optimizers.py is a modified version of Keras' default optimizer module (https://github.com/fchollet/keras/blob/master/keras/optimizers.py) that includes triangular CLR.

clr_optimizer_tests.ipynb contains tests demonstrating desired behavior.
