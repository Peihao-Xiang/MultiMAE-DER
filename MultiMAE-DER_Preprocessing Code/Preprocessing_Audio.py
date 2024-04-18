import os, warnings
import cv2
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from decord import VideoReader
from moviepy.editor import AudioFileClip

from scipy.io import wavfile # scipy library to read wav files
import numpy as np
from scipy.fftpack import dct
from matplotlib import pyplot as plt
from PIL import Image

input_size = 224
mid_point = 17

def normalize_audio(audio):
    audio = audio / np.max(np.abs(audio))
    return audio

def MFCC(signal,sample_rate):
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    frame_size = 0.025
    frame_stride = 0.0001

    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames *= np.hamming(frame_length)
    NFFT = 512

    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
    nfilt = 40

    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    num_ceps = 13
    mfcc = dct(filter_banks, type = 2, axis=1, norm="ortho")[:,1: (num_ceps + 1)] # keep 2-13
    cep_lifter = 22
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n/ cep_lifter)
    mfcc *= lift
    return mfcc

def preprocessing_audio(path, save_path):
    n = 1

    for class_name in os.listdir(path):
        class_dir = os.path.join(path, class_name)
        save_dir = os.path.join(save_path, class_name)
        
        for video_file in os.listdir(class_dir):
            video_path = os.path.join(class_dir, video_file)

            video_name = os.path.basename(video_path).split(".")[0]
            mp4_name = str(n) +  '.mp4'
            path_video_save = os.path.join(save_dir, mp4_name)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video = cv2.VideoWriter(path_video_save, fourcc, 16.0, (224, 224))
              
            audio_clip = AudioFileClip(video_path)
            audio_name = os.path.basename(video_path).split(".")[0]
            wave_name = str(audio_name) +  '.wav'
            path_audio_save = os.path.join('Data\\RAVDESS\\RAVDESS_WAVE', wave_name)

            audio_clip.write_audiofile(path_audio_save)
            fs, Audiodata = wavfile.read(path_audio_save)
            Audiodata = normalize_audio(Audiodata)
            step=int((len(Audiodata))/mid_point) - 1
            tx=np.arange(0,len(Audiodata),step)
            
            # Only Save Audio Spectrograms
            for i in range(16):
                signal=Audiodata[tx[i]:tx[i+2]]
                mfcc=MFCC(signal,fs)

                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
                cax = ax.matshow(
                    np.transpose(mfcc),
                    interpolation="nearest",
                    aspect="auto",
                    # cmap=plt.cm.afmhot_r,
                    origin="lower",
                )

                plt.axis('off')
                fig.savefig("MFCC.jpg")
                audio_img = Image.open("MFCC.jpg")
                audio_img = audio_img.resize((224, 224))
                audio_img = np.array(audio_img)

                cv2.imwrite('audio_img.jpg',audio_img)
                audio_img = Image.open("audio_img.jpg")
                audio_img = audio_img.resize((224, 224))
                audio_img = np.array(audio_img)

                plt.close('all')
                output_video.write(audio_img)

            output_video.release()
            cv2.destroyAllWindows()
            n = n + 1

    return n

if __name__ == '__main__':
    
    shutil.rmtree("Data\\RAVDESS\\RAVDESS_WAVE")
    os.mkdir("Data\\RAVDESS\\RAVDESS_WAVE")
    
    path_train = 'Data\\RAVDESS\\RAVDESS_RAW\\train'
    save_train_path = 'Data\\RAVDESS\\RAVDESS_Audio\\train'

    # path_test = 'Data\\RAVDESS\\RAVDESS_RAW\\test'
    # save_test_path = 'Data\\RAVDESS\\RAVDESS_Audio\\test'

    # path_val = 'Data\\RAVDESS\\RAVDESS_RAW\\val'
    # save_val_path = 'Data\\RAVDESS\\RAVDESS_Audio\\val'

    n_train = preprocessing_audio(path_train, save_train_path)
    # n_test = preprocessing_audio(path_test, save_test_path)
    # n_val = preprocessing_audio(path_val, save_val_path)

    print(n_train)
    # print(n_test)
    # print(n_val)
