import numpy as np
scalar = np.array(5)
vector = np.array([1,2,3])
matrix = np.array([[1,2,3],[4,5,6]])
cube = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])

print(scalar.shape,scalar.ndim)#() 0
print(vector.shape,vector.ndim)#(3,) 1
print(matrix.shape,matrix.ndim)#2,3 2
print(cube.shape,cube.ndim)#2,2,2 3

zeros = np.zeros((3,4))
ones = np.ones((3,4))
arange = np.arange(0,13)
random = np.random.randn(3,4)
print(zeros.dtype)
print(ones.dtype)
print(arange.dtype)
print(random.dtype)

x=np.random.randn(4,3,28,28)
# 纸上：__4__ 张 __3__ 通道的 __28__×__28__ 图片
print(x.shape,x.ndim)
assert x.shape == (4,3,28,28) , x.shape

y=np.random.randn(32,128,768)
# 纸上：一批 _32___ 条文本，每条 ___128_ 个 token，每个 token __768__ 维
print(y.shape,y.ndim)
assert y.shape ==  (32,128,768), y.shape


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