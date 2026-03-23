import numpy as np
import time
from matplotlib import pyplot as plt


def fft(x):
    # 转换为数组
    x = np.asarray(x, dtype=np.complex128)
    N = len(x)

    # 检查是否为2的幂次
    if (N & (N - 1)) != 0:
        raise ValueError("不是2的幂次。")

    # 递归终止条件
    if N <= 1:
        return x

    # 分为偶数索引和奇数索引两部分
    even = fft(x[0::2])
    odd = fft(x[1::2])

    # 计算
    k = np.arange(N / 2)
    W = np.exp(-2j * np.pi * k / N)

    T = W * odd

    return np.concatenate([even + T, even - T])


# 暖机
_ = fft(np.random.random(16))
_ = np.fft.fft(np.random.random(16))

print("===性能测试与比对===")
print(
    f"{'数组大小(N)':<12} | {'自编程序耗时(μs)':<18} | {'系统程序耗时(μs)':<16} | {'最大绝对误差'}"
)
print("-" * 70)

n = []
t = []
for p in range(4, 16):
    # 创建测试样例
    N = 2**p

    x = np.random.random(N) + 1j * np.random.random(N)

    # 自编算法结果
    start_c = time.perf_counter()
    res_c = fft(x)
    end_c = time.perf_counter()
    time_c = (end_c - start_c) * 1e6

    # 系统算法结果
    start_l = time.perf_counter()
    res_l = np.fft.fft(x)
    end_l = time.perf_counter()
    time_l = (end_l - start_l) * 1e6

    # 比较结果
    error = np.max(np.abs(res_c - res_l))

    print(f"2^{p} = {N:<7} | {time_c:<18.2f} | {time_l:<16.2f} | {error:.2e}")
    n.append(N)
    t.append(time_c)

plt.figure(figsize=(10, 7))
plt.plot(n, t, marker="o", color="blue")
plt.xscale("log", base=2)
plt.yscale("log")
plt.title("Execution Time", fontsize=14)
plt.xlabel("Array Size (N)", fontsize=12)
plt.ylabel("Execution Time (Microseconds / μs)", fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.show()
