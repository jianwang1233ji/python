import numpy as np
import time
"""这是对比向量化和for循环进行向量点乘运算时间"""
a = np.random.rand(100000)
b = np.random.rand(100000)
tic = time.time()
c = np.dot(a,b)
toc = time.time()
print(c)
print("vocter time:"+str(1000*(toc-tic)))

c = 0
tic = time.time()
for i in range(100000):
    c += a[i]*b[i]
    pass
print(c)
toc = time.time()
print("loop time:"+str(1000*(toc-tic)))

A = np.array([[1,2,3,4],[4,3,2,1],[2,3,4,5]])
print(A)
sum = A.sum(axis=0)
print(sum)
per = A/sum*100
print(per)
print(per.sum(axis=0))
#声明向量的形状，确保正确
assert(per.shape ==(3,4))


