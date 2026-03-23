import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

file_path = ".\question\waveform.dat"

# 读取数据
data = np.loadtxt(file_path)
t = data[:, 0]
y = data[:, 1]

N = len(t)

# 计算采样频率
dt = t[1] - t[0]
fs = 1.0 / dt

Y = np.fft.fft(y)

# 计算对应的频率轴
freqs = np.fft.fftfreq(N, dt)

p_freqs = freqs[: N // 2]

magnitude = (2.0 / N) * np.abs(Y[: N // 2])

# 寻找主要频率成分
# 设定阈值
threshold = np.max(magnitude) * 0.1

p_indices, _ = find_peaks(magnitude, height=threshold)

main_freqs = p_freqs[p_indices]
main_mags = magnitude[p_indices]

# 绘图可视化
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# 原始时域波形
ax1.plot(t, y, color="b", linewidth=0.5)
ax1.set_title("Original Waveform (Time Domain)", fontsize=14)
ax1.set_xlabel("Time (s)", fontsize=12)
ax1.set_ylabel("Amplitude", fontsize=12)
ax1.grid(True, linestyle="--", alpha=0.6)

# 频域频谱图
ax2.plot(p_freqs, magnitude, color="r")
ax2.set_title("FFT Frequency Spectrum", fontsize=14)
ax2.set_xlabel("Frequency (Hz)", fontsize=12)
ax2.set_ylabel("Magnitude", fontsize=12)
ax2.grid(True, linestyle="--", alpha=0.6)

# 标出主要频率峰值
for f, m in zip(main_freqs, main_mags):
    ax2.plot(f, m, "ko")
    ax2.annotate(
        f"{f:.2f} Hz",
        xy=(f, m),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize=10,
        color="darkgreen",
    )
    ax2.set_xlim(0, max(main_freqs) * 1.5)

plt.tight_layout()
plt.show()
