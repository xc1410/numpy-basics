# 纸上预测 10 组，10/10。第 7 组报错签名 (n?,k),(k,m?)->(n?,m?)
# 10 组纸上预测，正确 __/ 10
# 错的那几组，一组一行，格式固定：
#   第 N 组：纸上写的 ___ / 实际 ___ / 错在哪条规则 ___
import numpy as np
print(np.arange(24).reshape(4, 6).shape)
print(np.arange(24).reshape(2, 3, 4).shape)
print(np.arange(24).reshape(6, -1).shape)
print(np.zeros((2, 3, 4)).ndim)
print((np.random.randn(3,4) @ np.random.randn(4,7)).shape)
print((np.random.randn(10,5) @ np.random.randn(5,1)).shape)
print(np.random.randn(2,3) @ np.random.randn(2,3))    
print((np.random.randn(32,128,768) @ np.random.randn(768,512)).shape)
print((np.random.randn(2,3) + np.random.randn(3,)).shape)
print((np.random.randn(32,1) + np.random.randn(1,32)).shape)