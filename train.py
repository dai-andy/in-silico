import os
import pickle
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import joblib
from torch.utils.data import TensorDataset, DataLoader, random_split
from models import LinearDecoder, MLPDecoder
from tqdm import tqdm
import matplotlib.pyplot as plt

data_dir = "/nlp/scr/quevedo"
fmri_tensor_file = "synth_all_fmri_features.pt"
whisper_tensor_file = "synth_all_whisper_features.pt"
bert_tensor_file = "synth_all_bert_fmri_features.pt"
whisper_reg_wt_file = "S3_whisper-large_wts_layer32.jbl"

whisper = torch.load(os.path.join(data_dir, whisper_tensor_file))
bert = torch.load(os.path.join(data_dir, bert_tensor_file))
wt = joblib.load(os.path.join(data_dir, whisper_reg_wt_file))
wt = torch.tensor(wt).float().cuda()
fmri = torch.load(os.path.join(data_dir, fmri_tensor_file))

# only use top 128 most predictive voxels
top_indices = torch.tensor(np.load("top128ind.npz")['arr_0']).cuda()

real_features_dir = "features_cnk0.1_ctx16.0"
anth_features_dir = "anth_features_cnk0.1_ctx16.0"
libasr_features_dir = "libasr_features_cnk0.1_ctx16.0"

# get names of audio files
anth_story_names = [
        ent.replace('.npz', '') for ent in os.listdir(os.path.join(data_dir, anth_features_dir, whisper_model, 'encoder.0'))]
lib_story_names = [
        ent.replace('.npz', '') for ent in os.listdir(os.path.join(data_dir, libasr_features_dir, whisper_model, 'encoder.0'))]

# split into train and val
fmri_train_keys = ['adollshouse', 'adventuresinsayingyes', 'alternateithicatom', 'avatar', 'buck', 'exorcism',
            'eyespy', 'fromboyhoodtofatherhood', 'hangtime', 'haveyoumethimyet', 'howtodraw', 'inamoment',]
synth_train_keys = anth_story_names + lib_story_names
val_keys = ["wheretheressmoke"]

train_whisper = torch.cat([whisper[k] for k in synth_train_keys if k in whisper])
train_bert_fmri = torch.cat([bert[k] for k in fmri_train_keys if k in whisper])
train_bert_synth= torch.cat([bert[k] for k in synth_train_keys if k in whisper])
train_fmri = torch.cat([fmri[k] for k in fmri_train_keys if k in whisper])
# train_set = torch.cat([train_whisper, train_fmri])

val_bert = torch.cat([bert[k] for k in val_keys])
val_fmri = torch.cat([fmri[k] for k in val_keys])

fmri_train_dataset = TensorDataset(train_fmri, train_bert_fmri)
synth_train_dataset = TensorDataset(train_whisper, train_bert_synth)
fmri_val_dataset = TensorDataset(val_fmri, val_bert)
# train_dataset = TensorDataset(train_set)

batch_size = 128
synth_train_loader = DataLoader(synth_train_dataset, batch_size=batch_size, shuffle=True)
fmri_train_loader = DataLoader(fmri_train_dataset, batch_size=batch_size, shuffle=True)
fmri_val_loader = DataLoader(fmri_val_dataset, batch_size=batch_size, shuffle=True)

decoder = LinearDecoder(512, train_bert_fmri.shape[-1]).cuda().float()
# decoder = MLPDecoder(512, train_bert_fmri.shape[-1], 2).cuda().float()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(decoder.parameters())

fig, ax = plt.subplots()
losses = []
val_losses = []
real_losses = []

print("STARTING FMRI TRAINING")
num_epochs = 14
for epoch in tqdm(range(num_epochs)):
    for i, (x, y) in enumerate(fmri_train_loader):
        batch_num = epoch * len(fmri_train_loader) + i
        decoder.train()
        optimizer.zero_grad()

        x = x.cuda().float()
        x = torch.index_select(x, 1, top_indices)
        y = y.cuda().float()
        y_pred = decoder(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        real_losses.append((batch_num, loss.item()))

        optimizer.step()

        if i % 32 == 0:
            decoder.eval()
            batch_val_losses = []
            for x, y in fmri_val_loader:
                x = x.cuda().float()
                x = torch.index_select(x, 1, top_indices)
                y = y.cuda().float()
                y_pred = decoder(x)
                val_loss = loss_fn(y_pred, y)
                batch_val_losses.append(val_loss.item())
            val_losses.append((batch_num, torch.tensor(batch_val_losses).mean().item()))
    print(f"epoch {epoch} val loss: {val_loss.item()}")

ax.plot(*zip(*real_losses), color="blue")
ax.plot(*zip(*val_losses), color="red")

print('STARTING TRAINING ON SYNTHETIC DATA')
synth_losses = []
synth_num_epochs = 0
for epoch in tqdm(range(synth_num_epochs)):
    for i, (x, y) in enumerate(synth_train_loader):
        batch_num = num_epochs * len(fmri_train_loader) + epoch * len(synth_train_loader) + i
        decoder.train()
        optimizer.zero_grad()

        x = x.cuda().float()
        x = x @ wt
        x = torch.index_select(x, 1, top_indices)
        y = y.cuda().float()
        y_pred = decoder(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        synth_losses.append((batch_num, loss.item()))

        optimizer.step()

        if i % 8 == 0:
            decoder.eval()
            batch_val_losses = []
            for x, y in fmri_val_loader:
                x = x.cuda().float()
                # x = x @ wt
                x = torch.index_select(x, 1, top_indices)
                y = y.cuda().float()
                y_pred = decoder(x)
                val_loss = loss_fn(y_pred, y)
                batch_val_losses.append(val_loss.item())
            val_losses.append((batch_num, torch.tensor(batch_val_losses).mean().item()))
    print(f"epoch {epoch} val loss: {val_loss.item()}")

ax.plot(*zip(*synth_losses), color="purple")
ax.plot(*zip(*val_losses), color="red")
fig.savefig("loss_only_real_14.png")

