import numpy as np
np.random.seed(0)
print(np.random.rand(3))          # [0.5488135  0.71518937 0.60276338]

print(np.random.rand(3))          # seed 只设了一次，这次抽到的是新的三个数
print(np.random.rand(2, 3).shape) # (2, 3)
np.random.seed(0)
print(np.random.randn(3))         # [1.76405235 0.40015721 0.97873798]