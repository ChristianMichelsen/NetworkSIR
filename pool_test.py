import os
import time


# from pathos.pools import ProcessPool
# pool = ProcessPool(nodes=4)

# # do a blocking map on the chosen function
# results = pool.map(pow, [1,2,3,4], [5,6,7,8])

# # do a non-blocking map, then extract the results from the iterator
# results = pool.imap(pow, [1,2,3,4], [5,6,7,8])
# print("...")
# results = list(results)

# # do an asynchronous map, then get the results
# results = pool.amap(pow, [1,2,3,4], [5,6,7,8])
# while not results.ready():
#     time.sleep(5); 
#     print(".", end=' ')

# results = results.get()



# #%%

# # instantiate and configure the worker pool
# from pathos.multiprocessing import ProcessPool
# pool = ProcessPool(nodes=4)

# # do a blocking map on the chosen function
# print(pool.map(pow, [1,2,3,4], [5,6,7,8]))

# # do a non-blocking map, then extract the results from the iterator
# results = pool.imap(pow, [1,2,3,4], [5,6,7,8])
# print("...")
# print(list(results))

# # do an asynchronous map, then get the results
# results = pool.amap(pow, [1,2,3,4], [5,6,7,8])
# while not results.ready():
#     time.sleep(5); 
#     print(".", end=' ')
# print(results.get())

# # do one item at a time, using a pipe
# print(pool.pipe(pow, 1, 5))
# print(pool.pipe(pow, 2, 6))

# # do one item at a time, using an asynchronous pipe
# result1 = pool.apipe(pow, 1, 5)
# result2 = pool.apipe(pow, 2, 6)
# print(result1.get())
# print(result2.get())

# #%%

# # from pathos.multiprocessing import ProcessPool as Pool
# from pathos.pools import ProcessPool as Pool

# class Tasks:

#     def process_some_task(self, item):
#         print("Processing...", item, "by pid:", os.getpid())

# tasks = Tasks()


# # if __name__ == "__main__":
# with Pool(4) as pool:
#     pool.map(tasks.process_some_task, range(10))


#%%

import pathos as pa
pa.helpers.cpu_count()

#%%




from pathos.pools import ProcessPool, ThreadPool
import logging
log = logging.getLogger(__name__)

class PMPExample(object):
    def __init__(self):
        self.cache = {}

    def compute(self, x):
        self.cache[x] = x ** 3
        return self.cache[x]

    def threadcompute(self, xs):
        pool = ThreadPool(4)
        results = pool.map(self.compute, xs)
        return results

    def processcompute(self, xs):
        pool = ProcessPool(4)
        results = pool.map(self.compute, xs)
        return results

def parcompute_example():
    dc = PMPExample()
    dc2 = PMPExample()
    dc3 = PMPExample()
    dc4 = PMPExample()

    n_datapoints = 100
    inp_data = range(n_datapoints)
    r1 = dc.threadcompute(inp_data)
    assert(len(dc.cache) == n_datapoints)

    r2 = dc2.processcompute(inp_data)
    assert(len(dc2.cache) == 0)
    assert(r1 == r2)

    r3 = ProcessPool(4).map(dc3.compute, inp_data)
    r4 = ThreadPool(4).map(dc4.compute, inp_data)
    ProcessPool.__state__.clear()
    ThreadPool.__state__.clear()
    assert(r4 == r3 == r2)
    assert(len(dc3.cache) == 0)
    assert(len(dc4.cache) == n_datapoints)

    log.info("Size of threadpooled class caches: {0}, {1}".format(len(dc.cache), len(dc4.cache)))
    log.info("Size of processpooled class caches: {0}, {1}".format(len(dc2.cache), len(dc3.cache)))

if __name__ == '__main__':
    logging.basicConfig()
    log.setLevel(logging.INFO)
    parcompute_example()