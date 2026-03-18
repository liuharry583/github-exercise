from matplotlib import pyplot as plt

cmap = plt.get_cmap("jet_r")  # 获取颜色映射表
color = cmap(0.5)  # 用一个[0,1]之间的数取色
plt.plot([1, 2, 3], [2, 3, 1], color=color)  # 使用这个颜色画图
plt.show()
