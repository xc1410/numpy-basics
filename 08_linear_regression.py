import numpy as np
print(np.arange(1,6).mean())
print(np.arange(1,10).reshape(3,3).mean(axis=1))

x = np.arange(1,6)*1.0
y = 2 * x + 1
print(x , y)

def compute_loss(weight,bias):
    pred = weight * x + bias
    return ((pred-y)**2).mean()

print(compute_loss(0.0, 0.0))  # 期待 57.0   = (9 + 25 + 49 + 81 + 121) / 5
assert np.allclose(compute_loss(2.0, 1.0), 0), compute_loss(2.0, 1.0)   # 真实答案处 loss 为 0

def compute_grads(weight, bias):
    pred = weight * x + bias
    err = pred - y
    grad_weight = (2*x*err).mean()       # 你推的结果，用 err 和 x 写，一行
    grad_bias = (2*err).mean()         # 一行
    return grad_weight, grad_bias


print(compute_grads(0.0, 0.0))    # 期待 (np.float64(-50.0), np.float64(-14.0))
print(compute_grads(1.5, -0.5))   # 期待 (np.float64(-20.0), np.float64(-6.0))
print(compute_grads(2.0, 1.0))    # 期待 (np.float64(0.0), np.float64(0.0))   最低点梯度为 0

eps = 1e-6
for weight, bias in [(0.0, 0.0), (1.5, -0.5)]:
    grad_weight, grad_bias = compute_grads(weight, bias)
    numeric_weight = (compute_loss(weight + eps, bias) - compute_loss(weight - eps, bias)) / (2 * eps)
    numeric_bias = (compute_loss(weight, bias + eps) - compute_loss(weight, bias - eps)) / (2 * eps)
    print(numeric_weight, grad_weight, numeric_bias, grad_bias)
    assert abs(numeric_weight - grad_weight) < 1e-4, (numeric_weight, grad_weight)
    assert abs(numeric_bias - grad_bias) < 1e-4, (numeric_bias, grad_bias)

for lr in [0.01, 0.05, 0.08, 0.09]:
    weight, bias = 0.0, 0.0
    for step in range(200):
        grad_weight, grad_bias = compute_grads(weight, bias)
        weight = weight - lr * grad_weight
        bias = bias - lr * grad_bias
    print(lr, compute_loss(weight, bias), weight, bias)

weight, bias = 0.0, 0.0
for step in range(2000):
    grad_weight, grad_bias = compute_grads(weight, bias)
    weight = weight - 0.05 * grad_weight
    bias = bias - 0.05 * grad_bias
print(weight, bias)        # 期待 2.0000000000000013 0.9999999999999956
assert np.allclose([weight, bias], [2, 1]), (weight, bias)