import os
import torch
import torchaudio
import numpy as np
import scipy.io.wavfile
from pydub import AudioSegment
from tacotron2.hparams import create_hparams
from tacotron2.model import Tacotron2
from tacotron2.text import text_to_sequence
from waveglow.denoiser import Denoiser


#  Load Tacotron 2 & WaveGlow Models

def load_models():
    tacotron2 = Tacotron2()
    tacotron2.load_state_dict(torch.load("tacotron2_statedict.pt", map_location="cpu"))
    tacotron2.eval()

    waveglow = torch.load("waveglow_256channels.pt", map_location="cpu")
    waveglow.eval()
    denoiser = Denoiser(waveglow)

    return tacotron2, waveglow, denoiser

tacotron2, waveglow, denoiser = load_models()


#  Convert Text to Speech

def text_to_speech(file_path):
    # Read the file
    with open(file_path, "r") as file:
        text = file.read().strip()

    if not text:
        print(f"Skipping empty file: {file_path}")
        return

    # Convert text to sequence
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long()

    # Generate mel spectrogram
    with torch.no_grad():
        mel_outputs, mel_outputs_postnet, _, _ = tacotron2.inference(sequence)

    # Generate audio waveform
    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
    
    audio = denoiser(audio, strength=0.01)[:, 0]

    # Save as WAV
    wav_file = file_path.replace(".txt", ".wav")
    mp3_file = file_path.replace(".txt", ".mp3")

    scipy.io.wavfile.write(wav_file, 22050, audio.cpu().numpy())

    # Convert WAV to MP3
    sound = AudioSegment.from_wav(wav_file)
    sound.export(mp3_file, format="mp3")

    print(f"Audio saved as {mp3_file}")


#  Process All Subject-Level-No Files

for file in os.listdir():
    if file.endswith(".txt") and "-" in file:  # Ensure it's in subject-level-no format
        text_to_speech(file)
