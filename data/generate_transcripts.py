import whisper
import os
from tqdm import tqdm
import pandas as pd

model = whisper.load_model("base")

audio_dir = "story_audio"
results = {"title": [], "transcript": []}
for wav_file in tqdm(os.listdir(audio_dir)):
    result = model.transcribe(os.path.join(audio_dir, wav_file))
    results["title"].append(wav_file)
    results["transcript"].append(result["text"])

results_df = pd.DataFrame.from_dict(results)
out_file = "transcripts.csv"
results_df.to_csv(out_file)
print(f"success! transcripts saved to {out_file}")
