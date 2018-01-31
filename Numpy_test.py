import numpy as np
a = np.array([10, 20, 30, 40])
b = np.arange(4)
print(np.max(a))

print(a, b)
print(a*b, a-b, a+b, "\n -------------")


### numpy 基础运算1
c = np.array([[1, 1], [0, 1]])
d = np.arange(4).reshape(2, 2)
print(c, "\n --------------")
print(d, "\n --------------")

c_ = c * d
c_dot = np.dot(c, d)
print(c_)
print(c_dot)

a1 = np.random.random((3, 6))  # 返回范围内的一个随机浮点数（0，1）
print(a1)
print(np.max(a1))
print(np.min(a1))
print(np.sum(a1), "\n"*3)

### numpy基础运算2
A = np.arange(2, 14).reshape(3, 4)
print("A:", A)
print("最小值索引", np.argmin(A))
print("最大值索引", np.argmax(A))
print("平均值", A.mean())
print("中位数", np.median(A))
print("非零数", A.nonzero())
print("转置矩阵", A.T)
print("split夹片分割（将3~9之外的数用3或者9替换）", np.clip(A, 3, 9))
print("axis=1时，对矩阵列进行计算，axis = 1 时，对矩阵列进行计算：", np.sum(A, axis=1))
print("axis=0时，对矩阵行进行计算，axis = 0 时，对矩阵行进行计算：", np.sum(A, axis=0))
print("*"*20)

####### numpy 索引
B = np.arange(4, 20).reshape(4, 4)
print("B : ", B)
print("第一行的所有数值：", B[:1, :], "\n")
print("第一列的所有数值：", B[:, :1], "\n")

for raw in B:
	print("迭代输出每一行：", raw)

for rank in B.T:
	print("迭代输出每一列（B.T是B的转置矩阵）:", rank)

for item in B.flat:
	print("迭代输出每个元素（B.flat将矩阵变为一位矩阵）：", item)

######  Numpy 数组合并
M = np.array([1, 1, 1])
N = np.array([2, 2, 2])
print("M :", M)
print("N :", N)
M_N_v = np.vstack((M, N))  # vertical stack  垂直
M_N_h = np.hstack((M, N))  # horizontal stack  水平
print("上下垂直合并：\n", M_N_v, M_N_h.shape)
print("左右合并：\n", M_N_h, M_N_h.shape)
# 增加纬度（行 或者 列 ）
print("增加行纬度后：\n", M[:, np.newaxis])

###### numpy 数组分割
A = np.arange(12).reshape((3, 4))
print("A :\n", A)

print("对矩阵进行--行--分割：\n", np.split(A, 3, axis=0))
print("对矩阵进行--列--分割：\n", np.split(A, 4, axis=1))

print("对矩阵进行不等量的分割方法：\n", np.array_split(A, 3, axis=1))  # split方法增加一个array

print("横向的分割：\n", np.vsplit(A, 3))  # 简单的方法
print("竖向的分割：\n", np.hsplit(A, 2))

##### NUmpy的赋值（copy 与 deep copy）
a = np.arange(4)
print("a : ", a)

#用"="号赋值，元素之间有关联，改动一者全部改变
b = a
c = b
d = c
print("a , b , c , d 值:", a, b, c, d)
a[0] = 88
print("更改a[0]后的 a , b , c , d 值:", a, b, c, d, "\n")

#用 copy函数赋值，不会建立关联关系 deep copy
a = np.arange(4)
b = a.copy()
print("a , b的值：", a, b)
a[0] = 99
print("更改a[0]后的 a , b 的值： ", a, b)
