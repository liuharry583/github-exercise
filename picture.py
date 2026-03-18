from matplotlib import pyplot as plt

# 创建画布
fig = plt.figure(
    figsize=(10, 20),  # x长, y长
    subplot_kw={"projection": "polar"},
)  # 设置极坐标图
ax1 = fig.add_subplot(
    2,
    1,
    1,  # 添加子图, 2行1列, 这是第1个子图
    projection="polar",
)  # 极坐标图

fig, axes = plt.subplots(
    2,
    1,  # 行数, 列数
    figsize=(10, 10),
    subplot_kw={"projection": "polar"},
)
ax1 = axes[0]  # 第1个子图
