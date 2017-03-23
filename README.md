# CLR

![Alt text](images/triangularDiag.png?raw=true "Title")

This repository includes a Keras callback to be used in training that allows implementation of cyclical learning rate policies, as detailed in this paper https://arxiv.org/abs/1506.01186.

A cyclical learning rate is a policy of learning rate adjustment that increases the learning rate off a base value in a cyclical nature. Typically the frequency of the cycle is constant, but the amplitude is often scaled dynamically at either each cycle or each mini-batch iteration.

`clr_callback.py` contains the callback class `CyclicLR()`.

This class includes 3 built-in CLR policies, `'triangular'`, `'triangular2'`, and `'exp_range'`, as detailed in the original paper. It also allows for custom amplitude scaling functions, enabling easy experimentation.

Arguments for this class include:
* `base_lr`: initial learning rate, which is the lower boundary in the cycle. This overrides optimizer `lr`. Default 0.001.
* `max_lr`: upper boundary in the cycle. Functionally, it defines the cycle amplitude (`max_lr` - `base_lr`). The lr at any cycle is the sum of `base_lr` and some scaling of the amplitude; therefore `max_lr` may not actually be reached depending on scaling function. Default 0.006.
* `step_size`: number of training iterations per half cycle. Authors suggest setting `step_size` 2-8 x training iterations in epoch. Default 2000.
* `mode`: one of `{'triangular', 'triangular2', 'exp_range'}`. Values correspond to policies detailed below. If `scale_fn` is not `None`, this argument is ignored. Default `'triangular'`.
* `gamma`: constant in `'exp_range'` scaling function, `gamma^(cycle iterations)`. Default 1.
* `scale_fn`: Custom scaling policy defined by a single argument lambda function, where `0 <= scale_fn(x) <= 1` for all `x >= 0`. `mode` parameter is ignored when this argument is used. Default `None`.
* `scale_mode`: `{'cycle', 'iterations'}`. Defines whether `scale_fn` is evaluated on cycle number or cycle iterations (training iterations since start of cycle). Default is 'cycle'.

**NOTE: `base_lr` overrides optimizer lr**

The general structure of the policy algorithm is:

```python
cycle = np.floor(1+iterations/(2*step_size))
x = np.abs(iterations/step_size - 2*cycle + 1)
lr= base_lr + (max_lr-lr)*np.maximum(0, (1-x))*scale_fn(x)
```
where `x` is either `iterations` or `cycle`, depending on `scale_mode`.

# Policies

## triangular

![Alt text](images/triangularDiag.png?raw=true "Title")


This method is a simple triangular cycle.

Basic algorithm:

```python
cycle = np.floor(1+iterations/(2*step_size))
x = np.abs(iterations/step_size - 2*cycle + 1)
lr = base_lr + (max_lr-lr)*np.maximum(0, (1-x))
```

Default triangular clr policy example:
```python
    clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                        step_size=2000.)
    model.fit(X_train, Y_train, callbacks=[clr])
``` 

Results:

![Alt text](images/triangular.png?raw=true "Title")

## triangular2

![Alt text](images/triangular2Diag.png?raw=true "Title")

This method is a triangular cycle that decreases the cycle amplitude by half after each period, while keeping the base lr constant. This is an example of scaling on cycle number.

Basic algorithm:

```python
cycle = np.floor(1+iterations/(2*step_size))
x = np.abs(iterations/step_size - 2*cycle + 1)
lr = base_lr + (max_lr-lr)*np.maximum(0, (1-x))/float(2**(cycle-1))
```

Default triangular clr policy example:

```python
    clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                        step_size=2000., mode='triangular2')
    model.fit(X_train, Y_train, callbacks=[clr])
``` 

Results:

![Alt text](images/triangular2.png?raw=true "Title")

## exp_range

![Alt text](images/exp_rangeDiag.png?raw=true "Title")

This method is a triangular cycle that scales the cycle amplitude by a factor `gamma**(iterations)`, while keeping the base lr constant. This is an example of scaling on iteration.

Basic algorithm:

```python
cycle = np.floor(1+iterations/(2*step_size))
x = np.abs(iterations/step_size - 2*cycle + 1)
lr= base_lr + (max_lr-lr)*np.maximum(0, (1-x))*gamma**(iterations)
```

Default triangular clr policy example:

```python
    clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                        step_size=2000., mode='exp_range',
                        gamma=0.99994)
    model.fit(X_train, Y_train, callbacks=[clr])
``` 

Results:

![Alt text](images/exp_range.png?raw=true "Title")

## Custom Cycle-Policy

This method is a triangular cycle that scales the cycle amplitude sinusoidally. This is an example of scaling on cycle.

Basic algorithm:

```python
cycle = np.floor(1+iterations/(2*step_size))
x = np.abs(iterations/step_size - 2*cycle + 1)
lr= base_lr + (max_lr-lr)*np.maximum(0, (1-x))*0.5*(1+np.sin(cycle*np.pi/2.))
```

Default custom cycle-policy example:
```python
    clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
    clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                        step_size=2000., scale_fn=clr_fn,
                        scale_mode='cycle')
    model.fit(X_train, Y_train, callbacks=[clr])
``` 

Results:

![Alt text](images/cycle.png?raw=true "Title")

## Custom Iteration-Policy

This method is a triangular cycle that scales the cycle amplitude as a function of the cycle iterations. This is an example of scaling on iteration.

Basic algorithm:

```python
cycle = np.floor(1+iterations/(2*step_size))
x = np.abs(iterations/step_size - 2*cycle + 1)
lr= base_lr + (max_lr-lr)*np.maximum(0, (1-x))*1/(5**(iterations*0.0001))
```

Default custom cycle-policy example:
```python
    clr_fn = lambda x: 1/(5**(x*0.0001))
    clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                        step_size=2000., scale_fn=clr_fn,
                        scale_mode='iterations')
    model.fit(X_train, Y_train, callbacks=[clr])
``` 

Results:

![Alt text](images/iterations.png?raw=true "Title")

## Changing/resetting Cycle

During training, you may wish to adjust your cycle parameters: 

```python
clr._reset(new_base_lr,
           new_max_lr,
           new_step_size)
```
Calling `_reset()` allows you to start a new cycle w/ new parameters. 

`_reset()` also sets the cycle iteration count to zero. If you are using a policy with dynamic amplitude scaling, this ensures the scaling function is reset as well.

If an argument is not not included in the function call, then the corresponding parameter is unchanged in the new cycle. As a consequence, calling 

```python
clr._reset()
```

simply resets the scaling function.

## Report

`CyclicLR()` keeps track of learning rates at specific training iterations. This is what generated the plots above. This list of `(lr, iteration)` tuples is stored in the `record` attribute.

Note: iterations in the record is the running training iterations; it is distinct from the cycle iterations and does not reset. This allows you to plot your learning rates over training iterations, even after you change/reset the cycle.

Example:

![Alt text](images/reset.png?raw=true "Title")

## Order of learning rate augmentation
Note that the clr callback updates the learning rate prior to any further learning rate adjustments as called for in a given optimizer.

## Choosing a suitable `max_lr`
TO-DO


clr_callback_tests.ipynb contains tests demonstrating desired behavior of optimizers.
