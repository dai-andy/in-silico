import os
import pickle
import torch
from tqdm import tqdm

data_dir = "/nlp/scr/quevedo"
story2whisper_file = "training_story2whisper.pkl"
story2bert_file = "training_story2bert.pkl"

with open(os.path.join(data_dir, story2whisper_file), "rb") as f:
    story2whisper = pickle.load(f)
with open(os.path.join(data_dir, story2bert_file), "rb") as f:
    story2bert = pickle.load(f)

all_whisper_features = []
all_bert_features = []

for story in tqdm(story2whisper):
    whisper_tensor = torch.tensor(story2whisper[story])
    bert_tensor = torch.tensor(story2bert[story])
    if whisper_tensor.shape[0] != bert_tensor.shape[0]:
        print(f"skipping {story} because whisper and bert time dimension didn't match...")
        continue
    all_whisper_features.append(whisper_tensor)
    all_bert_features.append(bert_tensor)

torch.save(all_whisper_features, os.path.join(data_dir, "all_whisper_features.pt"))
torch.save(all_bert_features, os.path.join(data_dir, "all_bert_features.pt"))


