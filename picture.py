from matplotlib import pyplot as plt

### 选取颜色
cmap = plt.get_cmap("jet_r")  # 获取颜色映射表
color = cmap(0.5)  # 选取一个[0,1]之间的数取色

### 法1：先创建画布后加图
fig = plt.figure(
    figsize=(10, 20),  # x长, y长
    subplot_kw={"projection": "polar"},  # 设置极坐标图
)
# 创建单图
ax = fig.add_subplot()
# 创建多图
ax1 = fig.add_subplot(
    2,
    1,
    1,  # 添加子图, 2行1列, 这是第1个子图
    projection="polar",  # 设置极坐标图
)

### 法2：创建画布时直接加图
# 创建单图
fig, ax1 = plt.subplots(
    figsize=(10, 7),  # x长, y长
)
# 创建多图
fig, axes = plt.subplots(
    2,
    2,  # 行数, 列数
    figsize=(10, 10),
    subplot_kw={"projection": "polar"},
)
ax1 = axes[0, 0]  # 第1个子图

### 共享x轴, 创建第二个y轴
ax2 = ax1.twinx()
