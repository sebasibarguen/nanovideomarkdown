import os
import json

from openai import OpenAI

AUDIO_FILE_PATH = "data/audio.mp3"
AUDIO_CHUNK_DIR = "data/audio_chunk"

FRAMES_DIR = "data/frames"

TRANSCRIPT_DIR = "data/transcripts"


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Read all audio paths inside AUDIO_CHUNK_DIR
audio_chunks = [
    os.path.join(AUDIO_CHUNK_DIR, audio_file)
    for audio_file in os.listdir(AUDIO_CHUNK_DIR)
]

# Transcribe each audio chunk
for audio_path in audio_chunks:
    audio_file = open(audio_path, "rb")
    transcript = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-1",
        response_format="verbose_json",
        timestamp_granularities=["word"]
    )


    # Save the transcript to a file
    transcript_file = os.path.join(
        TRANSCRIPT_DIR, os.path.basename(audio_path).replace(".mp3", ".json")
    )

    with open(transcript_file, "w") as f:
        f.write(transcript.model_dump_json())


# Now we can create a transcript + frames dataset.

transcript_files = [
    os.path.join(TRANSCRIPT_DIR, transcript_file)
    for transcript_file in os.listdir(TRANSCRIPT_DIR)
]
transcript_files.sort()
frames = [os.path.join(FRAMES_DIR, frame) for frame in os.listdir(FRAMES_DIR)]
frames.sort()

# Create a dataset with the transcript and frames,
# for each transcript, we will have a list of 60 * 10 frames.
dataset = []

frames_per_chunk = 60 * 10

for i, transcript_file in enumerate(transcript_files):

    start = i * frames_per_chunk
    end = (i + 1) * frames_per_chunk

    with open(transcript_file, "r") as f:
        transcript = json.load(f)

    current_frames = frames[start:end]

    dataset.append({"transcript": transcript, "frames": frames, "start": start, "end": end})


# Save the dataset to a file
with open("data/dataset.json", "w") as f:
    json.dump(dataset, f)

