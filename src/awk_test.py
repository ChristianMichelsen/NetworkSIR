import awkward1 as ak
from numba import njit

x = ak.Array([[0, 1, 2], [3, 4]])
y = ak.Array([[0.1, 1.1, 2.1], [3.1, 4.1]])


@njit
def foo(x):
    for xi in x:
        print(xi)

foo(x)


# @njit
# def foo(x, y):
#     for i in range(len(x)):
#         xi = x[i]
#         yi = y[i]
#         print(xi, yi)

# foo(x, y)


# @njit
# def foo2(x, y, N):
#     for i in range(N):
#         xi = x[i]
#         yi = y[i]
#         print(xi, yi)

# foo2(x, y, len(x))


# # @njit
# # def bar(x, y):
# #     for xi, yi in zip(x, y):
# #         print(xi, yi)

# # bar(x, y)
# #

# # ak.size(x, axis=0)