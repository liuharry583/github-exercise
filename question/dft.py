import numpy as np


def dft(x):
    # 转换为数组并设定数据类型
    x = np.array(x, dtype=np.complex128)

    # 构造 DFT 矩阵
    N = len(x)

    n = np.arange(N)
    k = n.reshape((N, 1))

    e = np.exp(-2j * np.pi * k * n / N)

    # 矩阵乘法
    X = np.dot(e, x) / N**0.5

    return X


# 创建测试样例
N = 50

array = np.random.random(N) + 1j * np.random.random(N)

print(f"测试样例为：\n{array}")
# 自编算法结果
result1 = dft(array)

print(f"自编算法结果为：\n{result1}")
# 系统算法结果
result2 = np.fft.fft(array, norm="ortho")

print(f"系统算法结果为：\n{result2}")
# 比较结果
m_error = np.max(np.abs(result1 - result2))
print(f"结果最大差值为：\n{m_error}")
