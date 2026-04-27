import numpy as np
import os
import moviepy.editor as mp
import speech_recognition as sr
import soundfile as sf
from embedding4bert import Embedding4BERT
from deep_translator import GoogleTranslator
from googletrans import Translator
import pandas as pd
from GGO import GGO
from Global_Vars import Global_Vars
from Model_GCN import Model_GCN
from Model_LSTM import Model_LSTM
from Model_RAN import Model_RAN
from Model_SA_AMNet import Model_SA_AMNet
from Objective_Function import objfun_cls
from PROPOSED import PROPOSED
from Plot_results import *
from keras.utils import to_categorical
from numpy import matlib
import cv2 as cv
from SOA import SOA
from Spectral_Features import density, rms, zcr
from Spectral_Flux import spectralFlux
from THDN import THDN
from scipy.signal import find_peaks
import librosa
from WHO import WHO
from WSO import WSO


def spectral_centroid(x, samplerate=44100):
    magnitudes = np.abs(np.fft.rfft(x))  # magnitudes of positive frequencies
    length = len(x)
    freqs = np.abs(np.fft.fftfreq(length, 1.0 / samplerate)[:length + 1])  # positive frequencies
    return np.sum(magnitudes * freqs[:len(magnitudes)]) / np.sum(magnitudes)  # return weighted mean


# Read the Dataset
an = 0
if an == 1:
    Dataset = './Datasets/ml-100k/'
    Audio_path = Dataset + 'Audio/'
    text_path = Dataset + 'Text/'
    video_path = Dataset + 'Video/'
    file_path = text_path + 'items.csv'
    CSV_File = pd.read_csv(file_path)

    columns_to_check = ['YT-Trailer ID', 'Summary', 'Rating']
    filtered_data = CSV_File.dropna(subset=columns_to_check)
    video_filename = filtered_data['YT-Trailer ID']
    Text_of_file = filtered_data['Summary']
    video_filename.dropna(inplace=True)
    video_filename = np.asarray(video_filename)
    tar = filtered_data['Rating']
    targ = np.asarray(tar).astype('int')

    count = 0
    Images = []
    sample_rates = []
    Target = []
    audio_data = []
    transcriptions = []
    recognizer = sr.Recognizer()
    for j in range(len(video_filename)):
        video_file = os.path.join(video_path, video_filename[j] + '.mp4')
        audio_file = os.path.join(Audio_path, video_filename[j] + '.wav')
        cap = cv.VideoCapture(video_file)
        # Check if video opened successfully
        if not cap.isOpened():
            print(f"Error opening video stream or file: {video_file}")
            continue
        else:
            # Extract audio from the video
            video = mp.VideoFileClip(video_file)
            video.audio.write_audiofile(audio_file, codec='pcm_s16le')  # Save as .wav format
            # Load the audio file and get audio data and sample rate
            audio_signal, sample_rate = sf.read(audio_file)

            # Convert audio to text
            with sr.AudioFile(audio_file) as source:
                audio = recognizer.record(source)  # Read the audio file
                try:
                    text = recognizer.recognize_google(audio)  # Use Google Web Speech API for transcription
                except sr.UnknownValueError:
                    # text = "Could not understand audio"
                    text = Text_of_file[j]
                except sr.RequestError:
                    # text = "Error with the transcription service"
                    text = Text_of_file[j]
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        # Calculate the indices for five evenly spaced frames
        frame_indices = [int(i * total_frames / 50) for i in range(50)]

        print(
            f"Processing {video_filename[j]} with {total_frames} frames, capturing frames at indices: {frame_indices}")

        # Read and capture frames at calculated indices
        for idx in frame_indices:
            cap.set(cv.CAP_PROP_POS_FRAMES, idx)  # Set the current frame position
            ret, frame = cap.read()  # Read the frame
            if ret:
                print(j, len(video_filename), count)

                audio_data.append(audio_signal)
                sample_rates.append(sample_rate)
                transcriptions.append(text)
                resized = cv.resize(frame, (512, 512))  # Resize frame to 64x64
                Images.append(resized)  # Append resized frame to list
                Target.append(targ[j])
                count += 1
            else:
                print(f"Failed to capture frame at index {idx} in {video_filename[j]}")

        cap.release()

    aud = []
    for i in range(len(audio_data)):
        data = audio_data[i]
        signals = np.reshape(data, (data.shape[0] * data.shape[1]))
        aud.append(signals)

    audio_datas = []
    target_length = 10000
    for signal in aud:
        signal_length = len(signal)
        start_index = (signal_length - target_length) // 2
        end_index = start_index + target_length
        audio_datas.append(signal[start_index:end_index])

    Images = np.asarray(Images)
    Target = np.asarray(Target)
    audio_datas = np.asarray(audio_datas)
    train_labels = to_categorical(Target-5, dtype="uint8")  # because it takes the no's as classes (5, 6, 7, 8)
    transcriptions = np.asarray(transcriptions)
    sample_rates = np.asarray(sample_rates)

    index = np.arange(len(Images))
    np.random.shuffle(index)
    Images = np.asarray(Images)
    Shuffled_Images = Images[index]
    Shuffled_Target = train_labels[index]
    Shuffled_Text = transcriptions[index]
    Shuffled_audio = audio_datas[index]
    Shuffled_sample_rate = sample_rates[index]

    np.save('Images.npy', Shuffled_Images)
    np.save('index.npy', index)
    np.save('Text.npy', Shuffled_Text)
    np.save('Audios.npy', Shuffled_audio)
    np.save('Sample_rate.npy', Shuffled_sample_rate)
    np.save('Targets.npy', Shuffled_Target)


# Bidirectional Encoder Representations from Transformers (BERT)
an = 0
if an == 1:
    prep = np.load('Text.npy', allow_pickle=True)
    BERT = []
    for i in range(prep.shape[0]):
        print(i, prep.shape[0])
        emb4bert = Embedding4BERT("bert-base-cased")
        translator = Translator(service_urls=['translate.google.com'])
        translated = GoogleTranslator(source='auto', target='en').translate(prep[i])
        # preprocess = (prep[i]).tostring()  # bert-base-uncased
        tokens, embeddings = emb4bert.extract_word_embeddings(translated)
        BERT.append(embeddings)
    bert_feat = []
    for j in range(len(BERT)):
        bert = BERT[j]
        data = np.reshape(bert, (bert.shape[0] * bert.shape[1]))
        bert_feat.append(data)
    Min = np.min([len(k) for k in bert_feat])
    Bert_feat = [k[:Min] for k in bert_feat]
    np.save('BERT.npy', Bert_feat)  # Save the BERT data

# 3 Spectral Feature Extraction from Audio Data
an = 0
if an == 1:
    Audios = np.load('Audios.npy', allow_pickle=True)
    Sample_rate = np.load('Sample_rate.npy', allow_pickle=True)
    spectral = []
    for i in range(len(Audios)):
        print(i, len(Audios))
        Audio = Audios[i]
        cetroid = spectral_centroid(Audio)
        Density = density(Audio)
        Flux = spectralFlux(Audio)
        zero_crossings = librosa.zero_crossings(Audio, pad=True)
        zero_crossing = sum(zero_crossings)
        peaks, _ = find_peaks(Audio, height=0)
        peak_amp = np.mean(peaks)
        Thdn = THDN(np.uint8(Audio), Sample_rate[i])
        RMS = rms(Audio)
        ZCR = zcr(Audio)
        roll_off = librosa.feature.spectral_rolloff(y=Audio, sr=Sample_rate[i])
        mfccs = librosa.feature.mfcc(y=Audio, sr=Sample_rate[i], n_mfcc=1)
        spec = [cetroid, Density, Flux, zero_crossing, peak_amp, Thdn, RMS, ZCR, roll_off[0, 0], mfccs[0, 0]]
        spectral.append(spec)
    np.save('Spectral.npy', np.asarray(spectral))

# Optimization for classification
an = 0
if an == 1:
    Images = np.load('Images.npy', allow_pickle=True)
    BERT = np.load('BERT.npy', allow_pickle=True)
    Spectral = np.load('Spectral.npy', allow_pickle=True)
    Target = np.load('Targets.npy', allow_pickle=True)
    Img_feat = np.reshape(Images, (Images.shape[0], Images.shape[1] * Images.shape[2] * Images.shape[3]))
    Feat = np.concatenate((Img_feat, BERT, Spectral), axis=1)
    Global_Vars.Feat = Feat
    Global_Vars.Feat_1 = Images
    Global_Vars.Feat_2 = BERT
    Global_Vars.Feat_3 = Spectral
    Global_Vars.Target = Target
    Npop = 10
    Chlen = 4  # hidden neuron count, Epoch in GCNN, hidden neuron count, Epoch in LSTM
    xmin = matlib.repmat(np.asarray([5, 5, 5, 5]), Npop, 1)
    xmax = matlib.repmat(np.asarray([255, 50, 255, 50]), Npop, 1)
    fname = objfun_cls
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 50

    print("WSO...")
    [bestfit1, fitness1, bestsol1, time1] = WSO(initsol, fname, xmin, xmax, Max_iter)  # WSO

    print("GGO...")
    [bestfit2, fitness2, bestsol2, time2] = GGO(initsol, fname, xmin, xmax, Max_iter)  # GSO

    print("WHO...")
    [bestfit3, fitness3, bestsol3, time3] = WHO(initsol, fname, xmin, xmax, Max_iter)  # WHO

    print("SOA...")
    [bestfit4, fitness4, bestsol4, time4] = SOA(initsol, fname, xmin, xmax, Max_iter)  # SOA

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # PROPOSED

    BestSol_CLS = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(),
                   bestsol5.squeeze()]
    fitness = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]

    np.save('Fitness.npy', np.asarray(fitness))
    np.save('BestSol.npy', np.asarray(BestSol_CLS))

# Classification
an = 0
if an == 1:
    Images = np.load('Images.npy', allow_pickle=True)
    BERT = np.load('BERT.npy', allow_pickle=True)
    Spectral = np.load('Spectral.npy', allow_pickle=True)
    Target = np.load('Targets.npy', allow_pickle=True)
    Img_feat = np.reshape(Images, (Images.shape[0], Images.shape[1] * Images.shape[2] * Images.shape[3]))
    Feat = np.concatenate((Img_feat, BERT, Spectral), axis=1)
    BestSol = np.load('BestSol.npy', allow_pickle=True)
    EVAL = []
    Batch_Size = [4, 16, 32, 64, 128]
    for BS in range(len(Batch_Size)):
        learnperc = round(Feat.shape[0] * 0.75)
        Train_Data = Feat[:learnperc, :]
        Train_Target = Target[:learnperc, :]
        Test_Data = Feat[learnperc:, :]
        Test_Target = Target[learnperc:, :]
        Eval = np.zeros((10, 25))
        for j in range(BestSol.shape[0]):
            print(BS, j)
            sol = np.round(BestSol[j, :]).astype(np.int16)
            Eval[j, :], pred0 = Model_SA_AMNet(Images, BERT, Spectral, Target, sol)
        Eval[5, :], pred1 = Model_RAN(Train_Data, Train_Target, Test_Data, Test_Target, BS=Batch_Size[BS])
        Eval[6, :], pred2 = Model_LSTM(Train_Data, Train_Target, Test_Data, Test_Target, BS=Batch_Size[BS])
        Eval[7, :], pred3 = Model_GCN(Train_Data, Train_Target, Test_Data, Test_Target, BS=Batch_Size[BS])
        Eval[8, :], pred4 = Model_SA_AMNet(Images, BERT, Spectral, Target, BS=Batch_Size[BS])
        Eval[9, :] = Eval[4, :]
        EVAL.append(Eval)
    np.save('Eval_all_BS.npy', np.asarray(EVAL))  # Save the Eval_all_BS


plot_convergence()
ROC_curve()
Plot_Confusion()
Plot_Batchsize()
Plot_Kfold()
Sample_images()

