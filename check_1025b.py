import numpy as np
scores = np.arange(1,10).reshape(3,3)
row_sum=scores.sum(axis=1,keepdims=True)
assert row_sum.shape == (3,1),row_sum.shape
result = scores/row_sum
assert result.shape == (3,3),result.shape
result_sum=result.sum(axis=1)
assert np.allclose(result_sum, 1), result_sum

# row_sum=scores.sum(axis=1)
# assert row_sum.shape == (3,),row_sum.shape
# result = scores/row_sum
# result_sum=result.sum(axis=1)
# assert np.allclose(result_sum, 1), result_sum