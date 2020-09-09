import numpy as np
import numba as nb
from numba import njit

matrix = [[0, 0, 0, 0], [5, 8, 0, 0], [0, 0, 3, 0], [0, 6, 0, 0]]


# m = len(matrix)
# n = len(matrix[0])

# V = []
# ROW_INDEX = [0] # ROW_INDEX matrix has N+1 rows
# COL_INDEX = []
# NNZ = 0

# for i in range(m):
#     for j in range(n):
#         if matrix[i][j] != 0:
#             V.append(matrix[i][j])
#             COL_INDEX.append(j)
#             NNZ += 1
#     ROW_INDEX.append(NNZ)

# print(V)
# print(COL_INDEX)
# print(ROW_INDEX)


# # extract row 1

# row = 1
# row_start = ROW_INDEX[row]
# row_end = ROW_INDEX[row+1]

# print(V[row_start:row_end])
# print(COL_INDEX[row_start:row_end])


class IncrementalCOOMatrix:
    def __init__(self, shape, dtype):

        if dtype is np.int32:
            type_flag = "i"
        elif dtype is np.int64:
            type_flag = "l"
        elif dtype is np.float32:
            type_flag = "f"
        elif dtype is np.float64:
            type_flag = "d"
        else:
            raise Exception("Dtype not supported.")

        self.dtype = dtype
        self.shape = shape

        self.rows = array.array("i")
        self.cols = array.array("i")
        self.data = array.array(type_flag)

    def append(self, i, j, v):

        m, n = self.shape

        if i >= m or j >= n:
            raise Exception("Index out of bounds")

        self.rows.append(i)
        self.cols.append(j)
        self.data.append(v)

    def tocoo(self):

        rows = np.frombuffer(self.rows, dtype=np.int32)
        cols = np.frombuffer(self.cols, dtype=np.int32)
        data = np.frombuffer(self.data, dtype=self.dtype)

        return sp.coo_matrix((data, (rows, cols)), shape=self.shape)

    def __len__(self):

        return len(self.data)
