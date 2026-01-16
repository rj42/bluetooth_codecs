import argparse
import numpy as np
import librosa
from scipy import signal

def load_audio(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    return y, sr

def find_start(y, threshold=0.001, sr=None):
    """Найти начало звука"""
    for i, sample in enumerate(y):
        if abs(sample) > threshold:
            return i
    return 0

def find_start(y, sr, threshold=0.01, window_ms=50):
    """Найти начало по RMS окна"""
    window = int(window_ms / 1000 * sr)
    for i in range(0, len(y) - window, window):
        rms = np.sqrt(np.mean(y[i:i+window]**2))
        if rms > threshold:
            return i
    return 0

def find_offset(y1, y2, sr, start1, chunk_len_sec=10):
    """
    Берём кусок из y1 начиная со start1
    Ищем его в y2
    Возвращаем позицию в y2
    """
    chunk_len = int(chunk_len_sec * sr)
    chunk = y1[start1:start1 + chunk_len]

    search_len = int(100 * sr)
    search_area = y2[:search_len]

    corr = signal.correlate(search_area, chunk, mode='valid')
    start2 = np.argmax(corr)

    return start2

def normalize(y):
    return y / np.max(np.abs(y))

def normalize_rms(y, eps=1e-12):
    y = y - np.mean(y)
    rms = np.sqrt(np.mean(y*y))
    return y / (rms + eps)

def nope(y):
    return y

normalize = normalize_rms
#normalize = normalize

def ms(samples, sr):
    return samples / sr * 1000

def compare(file1, file2, duration=100, skip=0, offset=0):
    print(f"Файл 1: {file1}")
    print(f"Файл 2: {file2}")
    print("-" * 40)

    y1, sr1 = load_audio(file1)
    y2, sr2 = load_audio(file2)

    if sr1 != sr2:
        print(f"ОШИБКА: разный sample rate ({sr1} vs {sr2})")
        return

    sr = sr1
    print(f"Sample rate: {sr} Hz")
    print(f"Длина файл 1: {len(y1)/sr:.2f} sec")
    print(f"Длина файл 2: {len(y2)/sr:.2f} sec")

    # Найти старт в файле 1
    start1 = find_start(y1, sr1) + sr1 * skip

    # Найти соответствующую точку в файле 2 через корреляцию
    start2 = find_offset(y1, y2, sr, start1) + offset
    start1 += offset

    print(f"Старт звука:")
    print(f"  Файл 1: {ms(start1, sr):.2f} ms ({start1} samples)")
    print(f"  Файл 2: {ms(start2, sr):.2f} ms ({start2} samples)")
    print(f"  Разница: {ms(abs(start1 - start2), sr):.2f} ms ({abs(start1 - start2)} samples)")

    # Обрезать от старта
    y1 = y1[start1:]
    y2 = y2[start2:]

    # Обрезать до нужной длины
    samples = int(duration * sr)
    min_len = min(len(y1), len(y2), samples)
    y1 = y1[:min_len]
    y2 = y2[:min_len]

    print(f"Анализируем: {min_len/sr:.2f} sec")

    y1 = normalize(y1)
    y2 = normalize(y2)

    # Метрики
    corr = np.corrcoef(y1, y2)[0, 1]
    diff = y1 - y2
    diff_rms = np.sqrt(np.mean(diff**2))

    # Спектры
    spec1 = np.abs(librosa.stft(y1))
    spec2 = np.abs(librosa.stft(y2))

    spec_avg = (spec1 + spec2) / 2
    spec_avg[spec_avg < 0.00001] = 0.00001
    spec_diff_matrix = np.abs(spec1 - spec2) / spec_avg

    freqs = librosa.fft_frequencies(sr=sr)

    def format_freq(f):
        if f >= 1000:
            return f"{f // 1000}k"
        return str(f)

    freq_bands = [
        (0,     50),
        (5,     100),
        (100,   200),
        (200,   1000),
        (1000,  2000),
        (2000,  4000),
        (4000,  6000),
        (6000,  8000),
        (8000,  10000),
        (8000,  12000),
        (12000, 14000),
        (12000, 16000),
        (16000, 18000),
        (18000, 20000),
    ]

    print("-" * 40)
    print("РЕЗУЛЬТАТЫ:")
    print(f"Корреляция: {corr:.6f}")
    print(f"RMS разницы: {diff_rms:.6f}")

    print("-" * 40)
    print("РАЗНИЦА ПО ЧАСТОТАМ:")
    for (low, high) in freq_bands:
        low_str = format_freq(low)
        high_str = format_freq(high)
        name = f"{low_str}-{high_str} Hz"
        mask = (freqs >= low) & (freqs < high)
        if np.any(mask):
            band_diff = np.mean(spec_diff_matrix[mask, :])
            pct = band_diff * 100
            bar = "▓" * int(min(pct, 100) / 5) + "░" * (20 - int(min(pct, 100) / 5))
            print(f"  {name:>14}: {pct:6.1f}% {bar}")

    print("-" * 40)
    if corr > 0.999:
        print("ВЫВОД: Идентичны")
    elif corr > 0.99:
        print("ВЫВОД: Почти идентичны")
    elif corr > 0.95:
        print("ВЫВОД: Очень похожи")
    elif corr > 0.9:
        print("ВЫВОД: Похожи")
    else:
        print("ВЫВОД: Разные")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Сравнение аудио')
    parser.add_argument('--first', required=True)
    parser.add_argument('--second', required=True)
    parser.add_argument('--duration', type=int, default=100)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--offset', type=int, default=0)
    args = parser.parse_args()
    compare(args.first, args.second, args.duration, skip=args.skip, offset=args.offset)
