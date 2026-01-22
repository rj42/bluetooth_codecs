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
    """Найти начало по RMS окна"""
    window = int(window_ms / 1000 * sr)
    for i in range(0, len(y) - window, window):
        rms = np.sqrt(np.mean(y[i:i+window]**2))
        if rms > threshold:
            return i
    return 0

def find_offset(y1, y2, sr, start1, chunk_len_sec=10):
    """Корреляция для синхронизации"""
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
    """Выравнивает два аудиофайла, возвращает синхронизированные массивы"""
    y1, sr1 = load_audio(file1)
    y2, sr2 = load_audio(file2)

    if sr1 != sr2:
        raise ValueError(f"Разный sample rate: {sr1} vs {sr2}")

    sr = sr1
    print(f"Sample rate: {sr} Hz")
    print(f"Длина: {len(y1)/sr:.2f}s vs {len(y2)/sr:.2f}s")

    # Найти старт
    start1 = find_start(y1, sr) + int(sr * skip)
    start2 = find_offset(y1, y2, sr, start1) + offset
    start1 += offset

    print(f"Старт: {start1/sr*1000:.1f}ms vs {start2/sr*1000:.1f}ms")
    print(f"Разница: {abs(start1-start2)/sr*1000:.1f}ms")

    # Обрезать и выровнять
    y1 = y1[start1:]
    y2 = y2[start2:]

    samples = int(duration * sr)
    min_len = min(len(y1), len(y2), samples)
    y1 = y1[:min_len]
    y2 = y2[:min_len]

    print(f"Итого: {min_len/sr:.2f}s")

    # Нормализация
    y1 = normalize_rms(y1)
    y2 = normalize_rms(y2)

    return y1, y2, sr

def build_spectrogram(y, sr, n_fft=2048, hop_length=512):
    """Возвращает спектрограмму в dB"""
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return S_db

def plot_comparison(y1, y2, sr, name1="File 1", name2="File 2", output="compare.png"):
    """Строит 3 спектрограммы: file1, file2, diff (R/G overlay)"""

    S1 = build_spectrogram(y1, sr)
    S2 = build_spectrogram(y2, sr)

    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # Спека 1
    librosa.display.specshow(S1, sr=sr, hop_length=512, x_axis='time', y_axis='log', ax=axes[0], cmap='magma')
    axes[0].set_title(name1, fontsize=14)
    axes[0].set_ylabel('Частота (Hz)')

    # Спека 2
    librosa.display.specshow(S2, sr=sr, hop_length=512, x_axis='time', y_axis='log', ax=axes[1], cmap='magma')
    axes[1].set_title(name2, fontsize=14)
    axes[1].set_ylabel('Частота (Hz)')

    # DIFF: Red/Green overlay
    # Нормализуем в 0-1
    S1_norm = (S1 - S1.min()) / (S1.max() - S1.min())
    S2_norm = (S2 - S2.min()) / (S2.max() - S2.min())

    # RGB: R=file1, G=file2, B=0
    rgb = np.zeros((*S1_norm.shape, 3))
    rgb[:, :, 0] = S1_norm  # Red = file1
    rgb[:, :, 1] = S2_norm  # Green = file2
    rgb[:, :, 2] = 0        # Blue = 0

    # Flip для правильной ориентации (низкие частоты внизу)
    axes[2].imshow(rgb, aspect='auto', origin='lower',
                   extent=[0, len(y1)/sr, 0, sr/2])
    axes[2].set_title(f'DIFF: Красный={name1}, Зелёный={name2}, Жёлтый=одинаково', fontsize=14)
    axes[2].set_ylabel('Частота (Hz)')
    axes[2].set_xlabel('Время (сек)')
    axes[2].set_yscale('symlog', linthresh=100)  # Лог шкала для частот

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Сохранено: {output}")
    plt.close()

def save_aligned_wav(y1, y2, sr, prefix="aligned"):
    """Сохраняет выровненные wav файлы"""
    sf.write(f"{prefix}_1.wav", y1, sr)
    sf.write(f"{prefix}_2.wav", y2, sr)
    print(f"Сохранено: {prefix}_1.wav, {prefix}_2.wav")

def main():
    parser = argparse.ArgumentParser(description='Сравнение спектрограмм двух аудиофайлов')
    parser.add_argument('file1', help='Первый аудиофайл')
    parser.add_argument('file2', help='Второй аудиофайл')
    parser.add_argument('-d', '--duration', type=float, default=120, help='Длительность (сек)')
    parser.add_argument('-s', '--skip', type=float, default=0, help='Пропустить от начала (сек)')
    parser.add_argument('-o', '--output', default='compare.png', help='Выходной файл')
    parser.add_argument('--name1', default=None, help='Название файла 1')
    parser.add_argument('--name2', default=None, help='Название файла 2')
    parser.add_argument('--save-wav', action='store_true', help='Сохранить выровненные wav')

    args = parser.parse_args()

    name1 = args.name1 or args.file1
    name2 = args.name2 or args.file2

    print("="*50)
    print("Выравнивание аудио...")
    print("="*50)

    y1, y2, sr = align_audio(args.file1, args.file2, args.duration, args.skip)

    if args.save_wav:
        save_aligned_wav(y1, y2, sr)

    print("\n" + "="*50)
    print("Построение спектрограмм...")
    print("="*50)

    plot_comparison(y1, y2, sr, name1, name2, args.output)

if __name__ == '__main__':
    main()
