import numpy as np
import pandas as pd


##### pandas 属性
s = pd.Series([1, 3, 6, np.nan, 44, 1])  # 序列

print("s 如下:")
print(s)

a = np.array([1, 3, 6, np.nan, 44, 1])
print("a 如下:")
print(pd.Series(a))

datas = pd.date_range('20180101', periods=6)
print("datas is :\n\n", datas)

df = pd.DataFrame(np.random.randn(6, 4), index=datas, columns=['a', 'b', 'c', 'd'])  # random.randn() 从标准正态分布返回样本（或样本）
print("df:\n", df, "\n", df.dtypes)
print("行的名字:", df.index)
print("列的名字:", df.columns)


### pandas 选择数据
print("**"*40)
df1 = pd.DataFrame(np.arange(24).reshape(6, 4), index=datas, columns=['a', 'b', 'c', 'd'])
print("df1:\n", df1)
## 简单的选择 列 或者 行
print("选择a列的两种方法:\n", df1.a, "\n", df1['a'])
print("df1[:3]['a']\n", df1[:3]['a'])

print("-loc"*20)
# select by label : loc
print("选取某行--df1.loc['20180101']：\n", df1.loc['20180101'])  # 选择20180101标签的所有列
print("选取某列--df.loc['20180101']：\n", df1.loc['20180101', ['a', 'b']])  # 选择20180101标签的a，b列

print("-iloc"*20)
# select by position : iloc
print(df1.iloc[:4, 1:3])  # 选取0~4行，1~3列的数据
print(df1.iloc[[1, 3, 5], :])  # 选取1，3，5行的所有列数据输出

print("-ix"*20)
# mixed selection : ix
print(df1.ix[:3,['a', 'b']])  # 选择0~3行的「a，b」列输出
print(df1.ix[2:4, 1:3])  # 选择2~4行，1~3列输出

print("-other"*15)
### Boolean Indexing
print("df1:\n")
print(df1)
print(df1[df1.a > 8])  # 选择第a列值大于8的行数输出
print(df1[df1.a <= 8])


print("-set value"*5)
### 设置值
dates = pd.date_range('20180101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
print("原始df的值：")
print(df)

df.iloc[2, 2] = 1111  # 根据标签或索引更改元素值
df.loc['20180101', 'B'] = 2222
print("根据标签或索引更改值：")
print(df)

df.B[df.A > 4] = 0  # 将B列的 A便签的值大于 4的元素全部设为0
print("根据条件更改值：")
print(df)

df['E'] = np.arange(6)
df['F'] = pd.Series([1, 2, 3, 4, 5, 6], index=dates)
print("新增一列E后的df为：")
print(df)

print("-NaN"*15)
### 处理缺失值
print("\ndf原始数据：")
print(df)
print("\n设置缺失值后矩阵：")
df.iloc[0, 2:4] = np.nan
df.iloc[1:3, 3] = np.nan
print(df)

x = df.dropna(
	axis=0,  # 0:对行进行操作。1:对列进行操作
	how='any'  # 'any':只要存在Nan就drop掉。'all':必须全部都是Nan才可以drop掉
	)
print("\ndf.dropna() 处理缺失值过后的矩阵：")
print(x)

print("\ndf.fillna() 用特定值处理缺失值过后的矩阵：")
print(df.fillna(value=0))  # 如果是将 NaN 的值用其他值代替, 比如代替成 0

print("\ndf.isnull() 判断矩阵对应元素是否为空值：")
print(df.isnull())

print(np.any(df.isnull())==True)  # 检测在数据中是否存在 NaN, 如果存在就返回 True

### pandas 合并 concat
print("**"*40)

### pandas处理多组数据的时候往往会要用到数据的合并处理,使用 concat是一种基本的合并方式.而且concat中有很多参数可以调整,合并成你想要的数据形式
#定义资料集
df1 = pd.DataFrame(np.ones((3, 4))*0, columns=['a', 'b', 'c', 'd'])
df2 = pd.DataFrame(np.ones((3, 4))*1, columns=['a', 'b', 'c', 'd'])
df3 = pd.DataFrame(np.ones((3, 4))*2, columns=['a', 'b', 'c', 'd'])

print("合并后的矩阵:")
res = pd.concat([df1, df2, df3], axis=0)
print(res)

print("\nignore_index()重置 index 后的矩阵:")
print(pd.concat([df1, df2, df3], axis=0, ignore_index=True))

#Join合并方式
#join='outer'为预设值，因此未设定任何参数时，函数默认join='outer'。
#此方式是依照column来做纵向合并，有相同的column上下合并在一起，其他独自的column个自成列，原本没有值的位置皆以NaN填充。

# 定义资料集
df1 = pd.DataFrame(np.ones((3, 4))*0, columns=['a', 'b', 'c', 'd'], index=[1, 2, 3])
df2 = pd.DataFrame(np.ones((3, 4))*1, columns=['b', 'c', 'd', 'e'], index=[2, 3, 4])

##纵向"外"合并df1与df2
res = pd.concat([df1, df2], axis=0, join='outer')
print("\njoin outer 方式合并后的矩阵：")
print(res)

# 原理同上个例子的说明，但只有相同的column合并在一起，其他的会被抛弃
res = pd.concat([df1, df2], axis=0, join='inner')
print("\njoin inner 方式合并后的矩阵：")
print(res)

print("-plot"*15)
import matplotlib.pyplot as plt

data = pd.Series(np.random.randn(100), index=np.arange(100))
# data.cumsum()
# data.plot()
# plt.show()

data = pd.DataFrame(np.random.randn(100, 4), index=np.arange(100), columns=list("ABCD"))
print("data:")
print(data)
# data.plot()
# plt.show()

# 除了plot，我经常会用到还有scatter，这个会显示散点图，首先给大家说一下在 pandas 中有多少种方法
#     bar
#     hist
#     box
#     kde
#     area
#     scatter
#     hexbin
# 但是我们今天不会一一介绍，主要说一下 plot 和 scatter. 因为scatter只有x，y两个属性，我们我们就可以分别给x, y指定数据
ax = data.plot.scatter(x='A', y='B', color='DarkGreen', label='class1')
data.plot.scatter(x='A', y='C', color='DarkBlue', label='class2', ax=ax)
plt.show()
