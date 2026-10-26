import numpy as np
scores = np.arange(1, 10).reshape(3, 3)
print(scores[[0,2]])
print(scores[:,[0,2]])

grid = np.arange(1,10).reshape(3,3)
grid[grid>6] = 0
print(grid)

backup = scores.copy()
backup[0,0]=100
print(scores[0,0])

top_bottom = scores[[0,2]]
assert top_bottom.shape ==(2,3),top_bottom.shape
print(scores[[2, 0]])
assert grid.sum() == 21, grid.sum() 

scores = np.arange(1, 10).reshape(3, 3)
alias = scores
alias[0, 0] = 100
print(scores[0, 0])

x=np.array([-2.0,-1.0,0.0,1.0,2.0])
out = x.copy()
out[out<0]=0
assert np.allclose(out, [0, 0, 0, 1, 2]), out
assert np.allclose(x, [-2, -1, 0, 1, 2]), x 

scores = np.arange(1, 10).reshape(3, 3)
print(scores[[0, 1]].shape)        # 第 1 组：shape（2，3）
print(scores[[1, 1, 1]].shape)     # 第 2 组：shape(3,3)
print(scores[:, [1]].shape)        # 第 3 组：shape(3,)
g = np.arange(6)
g[g % 2 == 0] = 0
print(g)                           # 第 4 组：具体值(0,1,0,3,0,5)
h = np.arange(4)
k = h
k[0] = 9
print(h)                           # 第 5 组：具体值[9,1,2,3]