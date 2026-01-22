#!/usr/bin/env python3
import argparse
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy import signal
import soundfile as sf

def load_audio(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    return y, sr

def find_start(y, sr, threshold=0.01, window_ms=50):
    window = int(window_ms / 1000 * sr)
    for i in range(0, len(y) - window, window):
        rms = np.sqrt(np.mean(y[i:i+window]**2))
        if rms > threshold:
            return i
    return 0

def find_offset(y1, y2, sr, start1, chunk_len_sec=10):
    chunk_len = int(chunk_len_sec * sr)
    chunk = y1[start1:start1 + chunk_len]
    search_len = int(100 * sr)
    search_area = y2[:search_len]
    corr = signal.correlate(search_area, chunk, mode='valid')
    return np.argmax(corr)

def normalize_rms(y, eps=1e-12):
    y = y - np.mean(y)
    rms = np.sqrt(np.mean(y*y))
    return y / (rms + eps)

def align_audio(file1, file2, duration=120, skip=0, offset=0):
    y1, sr1 = load_audio(file1)
    y2, sr2 = load_audio(file2)

    if sr1 != sr2:
        raise ValueError(f"Разный sample rate: {sr1} vs {sr2}")

    sr = sr1
    print(f"Sample rate: {sr} Hz (max freq: {sr//2} Hz)")
    print(f"Длина: {len(y1)/sr:.2f}s vs {len(y2)/sr:.2f}s")

    start1 = find_start(y1, sr) + int(sr * skip)
    start2 = find_offset(y1, y2, sr, start1) + offset
    start1 += offset

    print(f"Старт: {start1/sr*1000:.1f}ms vs {start2/sr*1000:.1f}ms")

    y1 = y1[start1:]
    y2 = y2[start2:]

    samples = int(duration * sr)
    min_len = min(len(y1), len(y2), samples)
    y1 = y1[:min_len]
    y2 = y2[:min_len]

    print(f"Итого: {min_len/sr:.2f}s")

    y1 = normalize_rms(y1)
    y2 = normalize_rms(y2)

    return y1, y2, sr

def plot_comparison(y1, y2, sr, name1="File 1", name2="File 2", output="compare.png"):

    n_fft = 4096
    hop_length = 512

    # Спектрограммы в dB
    S1 = np.abs(librosa.stft(y1, n_fft=n_fft, hop_length=hop_length))
    S2 = np.abs(librosa.stft(y2, n_fft=n_fft, hop_length=hop_length))

    S1_db = librosa.amplitude_to_db(S1, ref=np.max)
    S2_db = librosa.amplitude_to_db(S2, ref=np.max)

    # DIFF в dB
    diff_db = S1_db - S2_db

    # Частоты для оси Y
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.times_like(S1_db, sr=sr, hop_length=hop_length)

    fig, axes = plt.subplots(3, 1, figsize=(18, 14))

    vmin, vmax = -80, 0  # dB range

    # Спека 1
    img1 = axes[0].pcolormesh(times, freqs, S1_db, shading='auto', cmap='magma', vmin=vmin, vmax=vmax)
    axes[0].set_title(name1, fontsize=14)
    axes[0].set_ylabel('Частота (Hz)')
    axes[0].set_ylim(0, sr // 2)  # До Nyquist!
    axes[0].set_yscale('symlog', linthresh=1000, linscale=0.5)
    plt.colorbar(img1, ax=axes[0], label='dB')

    # Спека 2
    img2 = axes[1].pcolormesh(times, freqs, S2_db, shading='auto', cmap='magma', vmin=vmin, vmax=vmax)
    axes[1].set_title(name2, fontsize=14)
    axes[1].set_ylabel('Частота (Hz)')
    axes[1].set_ylim(0, sr // 2)
    axes[1].set_yscale('symlog', linthresh=1000, linscale=0.5)
    plt.colorbar(img2, ax=axes[1], label='dB')

    # DIFF: diverging colormap
    diff_max = 10 # max(abs(diff_db.min()), abs(diff_db.max()), 20)
    img3 = axes[2].pcolormesh(times, freqs, diff_db, shading='auto', cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
    axes[2].set_title(f'DIFF: {name1} − {name2} (красный = {name1} громче, синий = {name2} громче)', fontsize=14)
    axes[2].set_ylabel('Частота (Hz)')
    axes[2].set_xlabel('Время (сек)')
    axes[2].set_ylim(0, sr // 2)
    axes[2].set_yscale('symlog', linthresh=1000, linscale=0.5)
    plt.colorbar(img3, ax=axes[2], label='ΔdB')

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Сохранено: {output}")
    plt.close()

    # Статистика diff
    print(f"\nСтатистика DIFF:")
    print(f"  Max diff: {diff_db.max():.2f} dB ({name1} громче)")
    print(f"  Min diff: {diff_db.min():.2f} dB ({name2} громче)")
    print(f"  Mean |diff|: {np.mean(np.abs(diff_db)):.2f} dB")
    print(f"  RMS diff: {np.sqrt(np.mean(diff_db**2)):.2f} dB")

def main():
    parser = argparse.ArgumentParser(description='DIFF спектрограмм')
    parser.add_argument('file1', help='Первый аудиофайл')
    parser.add_argument('file2', help='Второй аудиофайл')
    parser.add_argument('-d', '--duration', type=float, default=120, help='Длительность (сек)')
    parser.add_argument('-s', '--skip', type=float, default=0, help='Пропустить (сек)')
    parser.add_argument('-o', '--output', default='compare.png', help='Выходной файл')
    parser.add_argument('--name1', default=None, help='Название 1')
    parser.add_argument('--name2', default=None, help='Название 2')

    args = parser.parse_args()

    name1 = args.name1 or args.file1
    name2 = args.name2 or args.file2

    y1, y2, sr = align_audio(args.file1, args.file2, args.duration, args.skip)
    plot_comparison(y1, y2, sr, name1, name2, args.output)

if __name__ == '__main__':
    main()
