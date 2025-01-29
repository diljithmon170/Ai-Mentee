







import os
from transformers import pipeline
from pydub import AudioSegment
import torch
from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip
import pysrt

def generate_video_from_subject_level(subject_name,level):
    

    # Define text and audio file paths based on the subject_name
    text_file_path = f"{subject_name}-{level}.txt"  # Text file with subject-specific content
    audio_file_path = f"{subject_name}-{level}.mp3"  # Audio file with subject-specific content

    # Check if both text and audio files exist
    if not os.path.exists(text_file_path):
       print(f"Error: Text file '{subject_name}-{level}.txt' not found.")
       return
    if not os.path.exists(audio_file_path):
        print(f"Error: Text file '{subject_name}-{level}.mp3' not found.")
        return

    # Read the text from the file
    with open(f"{subject_name}-{level}.txt", "r", encoding="utf-8") as file:
        text = file.read()

    # Initialize the text-to-video model (CogVideoX-5B)
    model = pipeline("text-to-video", model="cog-video/cogvideo-5b", use_auth_token="your_token")

    # Generate video using the text file content
    
    video = model(text)

    # Save the video file (this will depend on the model's output format, e.g., mp4, avi, etc.)
    video_file_path = os.path.join( f"{subject_name}-{level}.mp4")
    with open(video_file_path, "wb") as video_file:
        video_file.write(video)

    # Now, process the audio
    
    audio = AudioSegment.from_file(f"{subject_name}-{level}.mp3")
    
    #mp3 to wav
    audio_file_path_wav = os.path.join( f"{subject_name}-{level}.wav")
    audio.export(audio_file_path_wav, format="wav")

    # Combine audio with the generated video (you may need an external library for this, like MoviePy)
    
    video_clip = VideoFileClip(video_file_path)
    audio_clip = AudioFileClip(audio_file_path_wav)
    
    # Set the audio to the video and write the final output
    final_video_path = os.path.join(f"{subject_name}-{level}.mp4")
    final_video = video_clip.set_audio(audio_clip)
    final_video.write_videofile(final_video_path, codec="libx264")

    
    
    # Generate Subtitles using Whisper
    generate_subtitles(audio_file_path_wav)
    # Add subtitles to video
    add_subtitles_to_video(final_video_path)


def generate_subtitles():
    

    # Initialize the Whisper model for ASR (Automatic Speech Recognition)
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large")

    # Read the audio file
    
    audio = AudioSegment.from_file()

    # Save a temporary WAV file for the transcription
    audio_file_path_wav = os.path.join( f"{subject_name}-{level}.wav")
    audio.export(audio_file_path_wav, format="wav")

    # Perform speech-to-text transcription
    
    transcription = asr(audio_file_path_wav)
    transcribed_text = transcription['text']
    

    # Segment the transcription into subtitles (you can adjust the segment length and overlap)
    
    subtitles = []
    words = transcribed_text.split(" ")
    segment_length = 10  # Length of each subtitle segment in seconds

    # Create segments based on segment_length
    for i in range(0, len(words), segment_length):
        start_time = i * 1  # seconds
        end_time = (i + segment_length) * 1  # seconds
        subtitle_text = " ".join(words[i:i + segment_length])
        
        # Create a subtitle entry (using pysrt)
        subtitle = pysrt.SubRipItem(index=len(subtitles) + 1, 
                                    start=pysrt.SubRipTime(seconds=start_time),
                                    end=pysrt.SubRipTime(seconds=end_time),
                                    text=subtitle_text)
        subtitles.append(subtitle)

    # Save subtitles to .srt file
    subtitles_file_path = os.path.join( f"{subject_name}-{level}-subtitles.srt")
    subtitles_file = pysrt.SubRipFile(subtitles)
    subtitles_file.save(subtitles_file_path)

    print(f"Subtitles saved to: {subtitles_file_path}")


def add_subtitles_to_video(video_file_path, subject_name=""):
    # Load the generated video and subtitles
    video_clip = VideoFileClip(video_file_path)
    subtitles_file_path = os.path.join(output_folder, f"{subject_name}_subtitles.srt")
    
    # Read the subtitles
    subs = pysrt.open(subtitles_file_path)
    
    # Create subtitle clips for each subtitle item
    subtitle_clips = []
    for sub in subs:
        # Create a TextClip for each subtitle
        subtitle_text_clip = TextClip(sub.text, fontsize=24, color='white', bg_color='black', size=video_clip.size)
        subtitle_text_clip = subtitle_text_clip.set_start(sub.start.seconds).set_end(sub.end.seconds)
        subtitle_text_clip = subtitle_text_clip.set_position(('center', 'bottom'))
        subtitle_clips.append(subtitle_text_clip)

    # Combine the video with subtitles
    final_video = CompositeVideoClip([video_clip] + subtitle_clips)
    
    # Write the final video with subtitles to file
    final_video_with_subtitles_path = os.path.join(f"{subject_name}-{level}.mp4")
    final_video.write_videofile(final_video_with_subtitles_path, codec="libx264")

    
subject_name=()
level=()



generate_video_from_subject_level(subject_name,level)
