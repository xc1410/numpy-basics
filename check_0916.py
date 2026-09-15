import numpy as np
x = np.random.randn(32,784)
assert x.shape == (32,784),x.shape
w = np.random.randn(784,128)
assert w.shape == (784,128),w.shape
b = np.random.randn(128,)
assert b.shape == (128,),b.shape
out = x@w+b
assert out.shape == (32,128),out.shape
out = np.maximum(out,0)
assert out.shape == (32, 128), out.shape