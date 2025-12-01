
# micrograd_musel

A tiny Autograd engine with a DAG that only operates over scalar values.
This was done by Andrej Karpathy, this is my recreation with some minor modifications to try it out. hanks to Andrej for his contribution! 

### Installation

```bash
pip install -i https://test.pypi.org/simple/ muselmicrograd
```

### Example usage

Below is a slightly contrived example showing a number of possible supported operations:

```python
from muselmicrograd.engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db
```


### Running tests

To run the unit tests you will have to install [PyTorch](https://pytorch.org/), which the tests use as a reference for verifying the correctness of the calculated gradients. Then simply:

```bash
python -m pytest
```
