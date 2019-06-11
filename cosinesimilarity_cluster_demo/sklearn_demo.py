import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

A = np.array([[0.1, 1.0, 1.0, 0.0],
              [1, 0.1, 0.0, 1.0, ],
              [1, 0.0, 0.0, 0.0, ],
              [0.0, 0.0, 1.0, 1.0,],
              [1.0, 1.0, 1.0, 1.0, ]]
             )
A_sparse = sparse.csr_matrix(A)

similarities = cosine_similarity(A_sparse)
print('pairwise dense output:\n {}\n'.format(similarities))

# also can output sparse matrices
similarities_sparse = cosine_similarity(A_sparse, dense_output=False)
print('pairwise sparse output:\n {}\n'.format(similarities_sparse))


from sklearn.metrics import pairwise_distances
A = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
 [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,]])
dist_out = 1 - pairwise_distances(A, metric="cosine")
print(dist_out)