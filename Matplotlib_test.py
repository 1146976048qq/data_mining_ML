import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("**基本用法**"*5)
x = np.linspace(-3, 3, 50)
y = x ** 2
# plt.plot(x, y)
# plt.show()

# figure() 显示
x = np.linspace(-10, 10, 50)
y1 = 3 * x + 2
y2 = x ** 2 + 2
plt.figure(num=1)
plt.plot(x, y1, label='y1', color='blue')
plt.plot(x, y2, color='red', linestyle='--', label='y2')
# plt.show()

#设置坐标轴参数
plt.xlim(-6, 6)
plt.ylim(-20, 50)
# plt.xlabel("X",)
# plt.ylabel("Y")

new_ticks = np.linspace(-10, 10, 10)
print("ticks:", new_ticks)
plt.xticks(new_ticks)
plt.yticks([5, 20, 35], [r'$bad\ \alpha$', r'$median\ \alpha$', r'$good\ \alpha$'])

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.legend(loc = 'best')

# 调整坐标轴位置
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))

ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))

# annotation 标注
x0 = 4
y0 = 3 * x0 + 2
plt.plot([x0, x0,], [0, y0,], 'k--')

plt.scatter([x0, ], [y0, ], s=40, color='Blue')  # 画出点（x0,y0）

plt.annotate(r'$3x+2=%s$' % y0, xy=(x0, y0), xycoords='data', xytext=(+30, -30),
             textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))  # 添加annotation注释

plt.text(0.6, -12.2, r'$This\ is\ the\ some\ text. \mu\ \sigma_i\ \alpha_t$',
         fontdict={'size': 14, 'color': 'b'})  # 添加text注释

plt.show()


print("**画图种类**"*5)

print("散点图：")
n = 512  # data size
X = np.random.normal(0, 1, n)  # 每一个点的 X值
Y = np.random.normal(0, 1, n)  # 每一个点的 Y值
T = np.arctan2(Y, X)  # for color value

plt.scatter(X, Y, s=75, c=T, name='散点图')
plt.xlim(-2.5, 2.5)
plt.xticks(())

plt.ylim(-2.5, 2.5)
plt.yticks(())

print("柱状图：")
n = 12
X = np.arange(n)
Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)

plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')

for x, y in zip(X, Y1):
    # ha: horizontal alignment
    # va: vertical alignment
    plt.text(x + 0.05, y + 0.05, '%.2f' % y, ha='center', va='bottom')

for x, y in zip(X, Y2):
    # ha: horizontal alignment
    # va: vertical alignment
    plt.text(x + 0.05, -y - 0.05, '%.2f' % y, ha='center', va='top')

plt.xlim(-.5, n)
plt.xticks(())
plt.ylim(-1.25, 1.25)
plt.yticks(())

print("等高线：")
def f(x, y):
    # the height function
    return (1 - x / 2 + x**5 + y**3) * np.exp(-x**2 - y**2)

n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
X, Y = np.meshgrid(x, y)

# use plt.contourf to filling contours
# X, Y and value for (X,Y) point
plt.contourf(X, Y, f(X, Y), 8, alpha=.75, cmap=plt.cm.hot)

# use plt.contour to add contour lines
C = plt.contour(X, Y, f(X, Y), 8, colors='black', linewidth=.5)
# adding label
plt.clabel(C, inline=True, fontsize=10)

plt.xticks(())
plt.yticks(())

print("image 图像:")
# image data
a = np.array([0.313660827978, 0.365348418405, 0.423733120134,
              0.365348418405, 0.439599930621, 0.525083754405,
              0.423733120134, 0.525083754405, 0.651536351379]).reshape(3,3)

"""
for the value of "interpolation", check this:
http://matplotlib.org/examples/images_contours_and_fields/interpolation_methods.html
for the value of "origin"= ['upper', 'lower'], check this:
http://matplotlib.org/examples/pylab_examples/image_origin.html
"""
plt.imshow(a, interpolation='None', cmap='bone', origin='lower')
plt.colorbar(shrink=.92)
plt.xticks(())
plt.yticks(())

plt.show()


print("**多图合并显示**"*5)
# example 1:
###############################
plt.figure(figsize=(6, 4))
# plt.subplot(n_rows, n_cols, plot_num)
plt.subplot(2, 2, 1)
plt.plot([0, 1], [0, 1])

plt.subplot(222)
plt.plot([0, 1], [0, 2])

plt.subplot(223)
plt.plot([0, 1], [0, 3])

plt.subplot(224)
plt.plot([0, 1], [0, 4])

plt.tight_layout()

# example 2:
###############################
plt.figure(figsize=(6, 4))
# plt.subplot(n_rows, n_cols, plot_num)
plt.subplot(2, 1, 1)
# figure splits into 2 rows, 1 col, plot to the 1st sub-fig
plt.plot([0, 1], [0, 1])

plt.subplot(234)
# figure splits into 2 rows, 3 col, plot to the 4th sub-fig
plt.plot([0, 1], [0, 2])

plt.subplot(235)
# figure splits into 2 rows, 3 col, plot to the 5th sub-fig
plt.plot([0, 1], [0, 3])

plt.subplot(236)
# figure splits into 2 rows, 3 col, plot to the 6th sub-fig
plt.plot([0, 1], [0, 4])


plt.tight_layout()
plt.show()

print("主次坐标轴：")
x = np.arange(0, 10, 0.1)
y1 = 0.05 * x ** 2
y2 = -1 * y1

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(x, y1, 'g-')
ax2.plot(x, y2, 'b--')

ax1.set_xlabel('X data')
ax1.set_ylabel('Y1 data', color='g')
ax2.set_ylabel('Y2 data', color='b')

plt.show()