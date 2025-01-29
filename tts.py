




import os
import torch
import soundfile as sf
from tacotron2 import Tacotron2
from waveglow import WaveGlow
from pydub import AudioSegment

def convert_text_files_to_mp3():
    try:
        # Load Tacotron 2 and WaveGlow models
        tacotron2 = Tacotron2()
        waveglow = WaveGlow()

        # Load pre-trained models
        tacotron2.load_state_dict(torch.load("tacotron2_statedict.pt"))  # Add the path to your Tacotron 2 checkpoint
        waveglow.load_state_dict(torch.load("waveglow_256channels_ljs_v2.pt"))  # Add path to WaveGlow checkpoint

        # Set models to evaluation mode
        tacotron2.eval()
        waveglow.eval()

        # Move models to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tacotron2 = tacotron2.to(device)
        waveglow = waveglow.to(device)

     

        # Loop through all text files
        for file_name in os.listdir():
            if file_name.endswith(".txt"):  # Only process .txt files
                
                output_audio = os.path.join(f"{os.path.splitext(file_name)[0]}.mp3")

                # Read text from file
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()

                # Preprocess the text 
                
                inputs = torch.tensor([list(text)], dtype=torch.long).to(device)

                # Generate Mel spectrogram from text using Tacotron 2
                mel_output, mel_length, alignment = tacotron2(inputs)

                # Use WaveGlow to generate audio from mel spectrogram
                audio = waveglow.infer(mel_output)

                # Convert audio tensor to numpy and save as WAV
                wav_output = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.wav")
                sf.write(wav_output, audio.cpu().numpy(), samplerate=22050)

                # Convert WAV to MP3 using pydub
                audio_segment = AudioSegment.from_wav(wav_output)
                audio_segment.export(output_audio, format="mp3")

                # Delete temporary WAV file
                os.remove(wav_output)

               

    except Exception as e:
        print("Error:", e)


