# Nano Video Markdown

Following [Karpathys call to action](https://twitter.com/karpathy/status/1760807877424640386). This repo takes in a video inside `./data/video.mp4` and generates markdown blog posts based on the video content.


## How to run

1. Insert video as `./data/video.mp4`
2. Run `python preprocess.py`, this extracts the audio, splits it into chunks and extracts frames.
3. Run `python transcribe.py`, this calls whisper to transcribe each audio chunk
4. Run `python main.py`, this then calls GPT4-V to generate the blog posts based on frames and transcription.


## Todos

- Add one last call to make all posts more "integrated"
- Improve how to sample frames
- Increase output tokens for GPT4-V
