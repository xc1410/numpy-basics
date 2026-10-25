for lr in [0.25,0.125,1.0,1.25]:
    print("lr=",lr)
    weight = 0.0
    for step in range(6):
        grad = 2 * (weight-4)
        weight = weight- lr*grad
        print(step,weight)

eps = 1e-6
weight = 1.0
numeric_grad = ((weight + eps - 4) ** 2 - (weight - eps - 4) ** 2) / (2 * eps)
analytic_grad = 2 * (weight - 4)
print(numeric_grad, analytic_grad)     # 期待 -6.000000000838668 -6.0
assert abs(numeric_grad - analytic_grad) < 1e-4, (numeric_grad, analytic_grad)