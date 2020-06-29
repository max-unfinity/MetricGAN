import os
import re
import numpy as np
import matplotlib.pyplot as plt
import librosa
from pathlib import Path
from tqdm.notebook import tqdm
from IPython import display

import torch
from torch.utils.data import Dataset

import itertools


def create_data(path, files_speech, files_noise, sample_rate, snr_levels, max_len=300, top_db=80.0, n_fft=512):
    '''
    Создает датасет микшированием каждого speech файла с каждым noise и на каждом snr_level.
    Итоговая длина датасета будет len(files_speech) * len(files_noise) * len(snr_levels).
    Строка датасета (1 файл) содержит магнитуды, фазы спектрограмм и wav файлы для clean_speech и noisy_speech. Т.е. всего 6 объектов: tuple(spec_noisy, spec_clean, phase_noisy, phase_clean, wave_noisy, wave_clean).
    Датасет удобнен и оптимален в этой работе. Файлы представляют из себя сериализованные бинарные .pt файлы, в которых хранится tuple из всего того, что может пригодится для обучения, восстановления спектрограмм и прослушивания.'''
    
    # [0]. preload all noise files
    # 1. load speech file
    # 2. pad speech and noise to wave_max_len (it will be exactly {max_len}, after converting to spectrogram)
    # 3. mix speech and noise matching SNR
    # 4. compute stft, make spectrogram in db scale
    # 5. convert np.array to torch.Tensor
    # 6. save on disk to path {path}

    noises_loaded = [librosa.load(p, sample_rate)[0] for p in files_noise]
    
    hop_length = n_fft//4  # default hop_length in librosa.stft
    wave_max_len = (max_len-1)*hop_length
    
    total = len(files_speech) * len(noises_loaded) * len(snr_levels)
    for i, (f_speech, noise, snr) in enumerate(tqdm(itertools.product(files_speech, noises_loaded, snr_levels), total=total)):
        
        clean, _ = librosa.load(f_speech, sample_rate)
        
        noise = librosa.util.fix_length(noise, wave_max_len)
        clean = librosa.util.fix_length(clean, wave_max_len)
        
        noisy = mix_with_snr(clean, noise, snr, random_shift=True)
        clean = normalize(clean)
        
        spec_noisy, phase_noisy = wave_to_spec(noisy, top_db=top_db, n_fft=n_fft)
        spec_clean, phase_clean = wave_to_spec(clean, top_db=top_db, n_fft=n_fft)
        
        data = tuple([torch.tensor(x) for x in (spec_noisy, spec_clean, phase_noisy, phase_clean, noisy, clean)])
        
        p = os.path.join(path, f'data_{i}.pt')
        if os.path.exists(p): raise FileExistsError(f'{p} is already exists')
        torch.save(data, p)

        
def wave_to_spec(wave, top_db=80.0, **stft_kwargs):
    '''Возвращает спектрограмму в дБ шкале (относительно сигнала с амплитудой 1.0) и её фазу. Также делается простая нормализация, которая не изменяет распределение данных, а только переводит в более удобный диапазон (около [-1, +1]). Нормализация здесь именно такая, чтобы не терять данные об относитльной громкости сигнала.'''
    stft = librosa.stft(wave, **stft_kwargs)
    magnitude, phase = np.abs(stft), np.angle(stft)
    spec = librosa.amplitude_to_db(magnitude, top_db=top_db)
    spec = spec/top_db  # normalized. spec.max() - spec.min() == 1.0
    return spec, phase

def spec_to_wave(spec_db, phase=None, top_db=80.0, **kwargs):
    '''Обратное преобразование wave_to_spec. Учитывается дБ шкала и нормализация.
    Если phase==None, то применяется алгоритм Гриффина-Лима.'''
    spec_db = np.array(spec_db)
    spec = spec_db*top_db
    spec = librosa.db_to_amplitude(spec)
    if phase is None:
        wave_rec = librosa.griffinlim(spec, **kwargs)
    else:
        phase = np.array(phase)
        spec_complex = spec * np.exp(phase*1j)
        wave_rec = librosa.istft(spec_complex, **kwargs)
    return wave_rec


def _true_sort_order(p):
    '''Возвращает key для human-like сортировки. Решает проблему, когда данные сортируются по типу:
    data_1, data_10, data_100, data_101 ...
    вместо ожидаемого:
    data_1, data_2, data_3 ...'''
    return [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]+|[0-9]+', str(p))]


class FileSpecDataset(Dataset):
    '''Обертка для датасета, созданного функцией create_data'''
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.files = list(Path(path).glob('*.pt'))
        assert len(self.files) > 0, f'no *.pt files in {path}'
        self.files.sort(key=_true_sort_order)
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, i):
        return torch.load(self.files[i])
    

def db(x, power=1):
    return power*10*np.log10(x)

def rms(amplitude):
    return np.sqrt(np.mean(amplitude**2))

def ms(amplitude):
    return np.mean(amplitude**2)


def padded(wav1, wav2, **kwargs):
    '''Добавляет паддинг более короткому сигналу, чтобы оба были одинаковой длинны.'''
    if len(wav1) < len(wav2):
        wav1 = np.pad(wav1, (0, len(wav2)-len(wav1)), **kwargs)
    else:
        wav2 = np.pad(wav2, (0, len(wav1)-len(wav2)), **kwargs)
    return wav1, wav2


def adjust_noise_to_match_snr(signal, noise, snr):
    '''Делает noise громче или тише - так, чтобы между signal и noise установилось Signal to Noise Ratio (SNR) равное snr.'''
    # вместо rms используется ms, чтобы лишний раз не возводить в степень и не извлекать корень от всего сигнала.
    rms_signal = ms(signal)
    rms_noise = ms(noise)
    
    rms_noise_matched = rms_signal/(10**(snr/10))
    k = rms_noise_matched/rms_noise
    
    return noise * np.sqrt(k)


def normalize(wave):
    '''Линейно нормализует громкость (амплитуду) сигнала до 1.0 by peak.'''
    return wave/np.abs(wave).max()


def mix_with_snr(signal, noise, snr, random_shift=True):
    '''Делает микс из двух сигналов (второй обычно шум). Их относительная громкость регулируется с помощью {snr}.'''
    noise = adjust_noise_to_match_snr(signal, noise, snr)
    mixed = normalize(signal + noise)
    return mixed


def process_batch_as_authors(specs_db, top_db=80.0):
    '''Переводит из dB-scale обратно в амлитуду. Делает нормализацию, как у авторов'''
    specs = 10.**(specs_db * top_db / 20.)
    mean = specs.mean(dim=2).unsqueeze(-1)
    std = specs.std(dim=2).unsqueeze(-1)

    specs_normalized = (specs-mean)/std
    return specs, specs_normalized

def spec_to_wave_amplitude(spec, phase=None, **kwargs):
    spec = np.array(spec)
    if phase is None:
        wave_rec = librosa.griffinlim(spec, **kwargs)
    else:
        phase = np.array(phase)
        spec_complex = spec * np.exp(phase*1j)
        wave_rec = librosa.istft(spec_complex, **kwargs)
    return wave_rec
