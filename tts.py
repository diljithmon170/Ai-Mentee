import os
import torch
import soundfile as sf
from pydub import AudioSegment
from tacotron2 import Tacotron2  # Ensure correct import
from waveglow import WaveGlow  # Ensure correct import
from transformers import AutoTokenizer

def convert_text_files_to_mp3(output_folder="output_audio"):
    try:
        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Load Tacotron 2 and WaveGlow models
        tacotron2 = Tacotron2()
        waveglow = WaveGlow()

        # Load pre-trained models (Provide correct paths)
        tacotron2.load_state_dict(torch.load("tacotron2_statedict.pt", map_location="cpu"))
        waveglow.load_state_dict(torch.load("waveglow_256channels_ljs_v2.pt", map_location="cpu"))

        # Set models to evaluation mode
        tacotron2.eval()
        waveglow.eval()

        # Move models to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tacotron2.to(device)
        waveglow.to(device)

        # Tokenizer for text preprocessing (Assuming Tacotron2 uses a tokenizer)
        tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")

        # Process all text files in the current directory
        for file_name in os.listdir():
            if file_name.endswith(".txt"):  # Only process .txt files
                file_path = os.path.join(os.getcwd(), file_name)

                # Read text from file
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read().strip()

                # Preprocess the text
                inputs = tokenizer(text, return_tensors="pt").input_ids.to(device)

                # Generate Mel spectrogram from text using Tacotron 2
                mel_output, _, _ = tacotron2(inputs)

                # Use WaveGlow to generate audio from mel spectrogram
                audio = waveglow.infer(mel_output)

                # Convert audio tensor to numpy and save as WAV
                wav_output = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.wav")
                sf.write(wav_output, audio.cpu().numpy(), samplerate=22050)

                # Convert WAV to MP3 using pydub
                mp3_output = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.mp3")
                audio_segment = AudioSegment.from_wav(wav_output)
                audio_segment.export(mp3_output, format="mp3")

                # Delete temporary WAV file
                os.remove(wav_output)

                print(f"Converted {file_name} to {mp3_output}")

    except Exception as e:
        print("Error:", e)

# Run the function
convert_text_files_to_mp3()
