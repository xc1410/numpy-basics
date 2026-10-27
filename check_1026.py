import numpy as np
x=np.arange(1,6)
y=2*x+1
def compute_loss(weight,bias):
    pred=x*weight+bias
    return ((pred-y)**2).mean()

def compute_grads(weight,bias):
    pred=x*weight+bias
    err=pred-y
    grad_weight=(2*err*x).mean()
    grad_bias=(2*err).mean()
    return (grad_weight,grad_bias)



eps = 1e-6
weight = 0
bias = 0
numeric_weight = (compute_loss(weight + eps, bias) - compute_loss(weight - eps, bias)) / (2 * eps)
numeric_bias = (compute_loss(weight, bias + eps) - compute_loss(weight, bias - eps)) / (2 * eps)
assert np.allclose([numeric_weight, numeric_bias], [-50,- 14]), [numeric_weight, numeric_bias]

x=np.array([-2.0,-1.0,0.0,1.0,2.0])
out = x.copy()
out[out<0]=0
assert np.allclose(out, [0, 0, 0, 1, 2]), out
assert np.allclose(x, [-2, -1, 0, 1, 2]), x 