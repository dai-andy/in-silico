#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
from ridge_utils.npp import zs
from ridge_utils.interpdata import lanczosinterp2D, simulate
from ridge_utils.util import make_delayed
import pickle
import torch
import joblib
from tqdm import tqdm

data_dir = "/nlp/scr/quevedo"
real_features_dir = "features_cnk0.1_ctx16.0"
anth_features_dir = "anth_features_cnk0.1_ctx16.0"
libasr_features_dir = "libasr_features_cnk0.1_ctx16.0"
whisper_model = "whisper-large"
wordseqs_file = "wordseqs.pkl"
fmri_file = "UTS03_responses.jbl"
story2fmri_file = "synth_training_story2fmri.pkl"
story2whisper_file = "synth_training_story2whisper.pkl"
story2snippets_file = "synth_training_story2snippets.pkl"


# In[2]:

'''
def get_interpolated_snippets(features, sincmat, all_snippets):
    i, j = sincmat.nonzero()
    midpoints = np.array([
        np.median(j[i == idx]) for idx in range(features.shape[0])
    ])
    midpoints[np.isnan(midpoints)] = -1
    indices = midpoints.astype(np.int32)
    snippets = np.array([
        all_snippets[indices[i]] if indices[i] != -1 else None for i in range(indices.shape[0])
    ])
    return snippets


# In[4]:


with open(os.path.join(data_dir, wordseqs_file), "rb") as f:
    wordseqs = pickle.load(f)
fmri = joblib.load(os.path.join(data_dir, fmri_file))


# In[6]:


story2fmri = {}
# mapping from stories to features
story2whisper = {}
# mapping from stories to sequences of text snippets (transcriptions)
story2snippets = {}

# map stories with real fmri
features_dir = real_features_dir
story_names = [
        ent.replace('.npz', '') for ent in os.listdir(os.path.join(data_dir, features_dir, whisper_model, 'encoder.0')) 
]

for story in tqdm(story_names):
    if not os.path.isfile(os.path.join(data_dir, features_dir, whisper_model, f"{story}_snippet_texts.npz")):
        print(f"Skipping {story} because snippet texts were not generated.")
        continue
    if story not in wordseqs:
        print(f"Skipping {story} because it wasn't found in wordseqs.")
        continue
    if story not in fmri:
        print(f"Skipping {story} because it wasn't found in fmri.")
        continue
    snippet_times = np.load(os.path.join(data_dir, features_dir, whisper_model, f"{story}_times.npz"))['times'][:,1] # shape: (time,)
    snippet_texts = np.load(os.path.join(data_dir, features_dir, whisper_model, f"{story}_snippet_texts.npz"))['snippet_texts'] # shape: (time,)

    snippet_features = np.load(os.path.join(data_dir, features_dir, whisper_model, f"{story}.npz"))['features'] # shape: (time, model dim.)
    # now that we have the word-by-word whisper features (snippet_features), we can downsample them to fMRI timings
    downsampled_features, sincmat = lanczosinterp2D(
        snippet_features, snippet_times,
        wordseqs[story].tr_times)
        # simulate(snippet_times[-1] // 2, offset=-9))
    trimmed_features = downsampled_features[10:-5]
    normalized_features = np.nan_to_num(zs(trimmed_features))
    delayed_features = make_delayed(normalized_features, range(1, 5))
    downsampled_snippets = get_interpolated_snippets(downsampled_features, sincmat, snippet_texts)
    trimmed_snippets = downsampled_snippets[10:-5]

    assert delayed_features.shape[0] == trimmed_snippets.shape[0]
    assert delayed_features.shape[0] == fmri[story].shape[0]

    story2fmri[story] = fmri[story]
    story2whisper[story] = delayed_features
    story2snippets[story] = trimmed_snippets

# get story names from anthropocene
features_dir = anth_features_dir
anth_story_names = [
        ent.replace('.npz', '') for ent in os.listdir(os.path.join(data_dir, features_dir, whisper_model, 'encoder.0')) 
]

# In[13]:
for story in tqdm(anth_story_names):
    if not os.path.isfile(os.path.join(data_dir, features_dir, whisper_model, f"{story}_snippet_texts.npz")):
        print(f"Skipping {story} because snippet texts were not generated.")
        continue
    # if story not in wordseqs:
    #    print(f"Skipping {story} because it wasn't found in wordseqs.")
    #    continue
    # if story not in fmri:
    #    print(f"Skipping {story} because it wasn't found in fmri.")
    #    continue
    snippet_times = np.load(os.path.join(data_dir, features_dir, whisper_model, f"{story}_times.npz"))['times'][:,1] # shape: (time,)
    snippet_texts = np.load(os.path.join(data_dir, features_dir, whisper_model, f"{story}_snippet_texts.npz"))['snippet_texts'] # shape: (time,)

    snippet_features = np.load(os.path.join(data_dir, features_dir, whisper_model, f"{story}.npz"))['features'] # shape: (time, model dim.)
    # now that we have the word-by-word whisper features (snippet_features), we can downsample them to fMRI timings
    downsampled_features, sincmat = lanczosinterp2D(
        snippet_features, snippet_times,
        simulate(snippet_times[-1] // 2, offset=-9))
    trimmed_features = downsampled_features[10:-5]
    normalized_features = np.nan_to_num(zs(trimmed_features))
    delayed_features = make_delayed(normalized_features, range(1, 5))
    downsampled_snippets = get_interpolated_snippets(downsampled_features, sincmat, snippet_texts)
    trimmed_snippets = downsampled_snippets[10:-5]

    assert delayed_features.shape[0] == trimmed_snippets.shape[0]
    # assert delayed_features.shape[0] == fmri[story].shape[0]

    # story2fmri[story] = fmri[story]
    story2whisper[story] = delayed_features
    story2snippets[story] = trimmed_snippets


# get story names from libasr
features_dir = libasr_features_dir
lib_story_names = [
        ent.replace('.npz', '') for ent in os.listdir(os.path.join(data_dir, features_dir, whisper_model, 'encoder.0')) 
]

for story in tqdm(lib_story_names):
    if not os.path.isfile(os.path.join(data_dir, features_dir, whisper_model, f"{story}_snippet_texts.npz")):
        print(f"Skipping {story} because snippet texts were not generated.")
        continue
    # if story not in wordseqs:
    #    print(f"Skipping {story} because it wasn't found in wordseqs.")
    #     continue
    # if story not in fmri:
    #     print(f"Skipping {story} because it wasn't found in fmri.")
    #    continue
    snippet_times = np.load(os.path.join(data_dir, features_dir, whisper_model, f"{story}_times.npz"))['times'][:,1] # shape: (time,)
    snippet_texts = np.load(os.path.join(data_dir, features_dir, whisper_model, f"{story}_snippet_texts.npz"))['snippet_texts'] # shape: (time,)

    snippet_features = np.load(os.path.join(data_dir, features_dir, whisper_model, f"{story}.npz"))['features'] # shape: (time, model dim.)
    # now that we have the word-by-word whisper features (snippet_features), we can downsample them to fMRI timings
    downsampled_features, sincmat = lanczosinterp2D(
        snippet_features, snippet_times,
        simulate(snippet_times[-1] // 2, offset=-9))
    trimmed_features = downsampled_features[10:-5]
    normalized_features = np.nan_to_num(zs(trimmed_features))
    delayed_features = make_delayed(normalized_features, range(1, 5))
    downsampled_snippets = get_interpolated_snippets(downsampled_features, sincmat, snippet_texts)
    trimmed_snippets = downsampled_snippets[10:-5]

    assert delayed_features.shape[0] == trimmed_snippets.shape[0]
    # assert delayed_features.shape[0] == fmri[story].shape[0]

    # story2fmri[story] = fmri[story]
    story2whisper[story] = delayed_features
    story2snippets[story] = trimmed_snippets
# In[9]:


delayed_features.shape


# In[11]:


trimmed_snippets.shape


# In[12]:


# fmri[story].shape


# In[14]:


print('length of fmri dictionary:', len(story2fmri))


# In[15]:


# with open(os.path.join(data_dir, story2fmri_file), "wb") as f:
#    pickle.dump(story2fmri, f)
# with open(os.path.join(data_dir, story2whisper_file), "wb") as f:
#    pickle.dump(story2whisper, f)
# with open(os.path.join(data_dir, story2snippets_file), "wb") as f:
#    pickle.dump(story2snippets, f)

breakpoint()
# In[16]:


story2whisper_tensor = {}
story2fmri_tensor = {}
story2bert_tensor = {}


# In[23]:


story2bert_file = "synth_training_story2bert.pkl"
with open(os.path.join(data_dir, story2bert_file), "rb") as f:
    story2bert = pickle.load(f)


# In[28]:

for story in tqdm(list(story2whisper.keys())):
    # if story not in story2fmri:
    #    print(f"skipping {story} because it wasn't found in fmri...")
    #    continue
    if story not in story2bert:
        print(f"skipping {story} because it wasn't found in bert...")
        continue
    bert_tensor = torch.tensor(story2bert[story])
    whisper_tensor = torch.tensor(story2whisper[story])
    if story in story2fmri:
        fmri_tensor = torch.tensor(story2fmri[story])
        if fmri_tensor.shape[0] != bert_tensor.shape[0]:
            continue
        # assert fmri_tensor.shape[0] == bert_tensor.shape[0]
        story2fmri_tensor[story] = fmri_tensor
    # if fmri_tensor.shape[0] != bert_tensor.shape[0]:
    #    print(f"skipping {story} because fmri and bert time dimension didn't match...")
    #    continue
    if whisper_tensor.shape[0] != bert_tensor.shape[0]:
        print(f"skipping {story} because whisper and bert time dimension didn't match...")
        continue
    assert whisper_tensor.shape[0] == bert_tensor.shape[0]
    story2whisper_tensor[story] = whisper_tensor
    story2bert_tensor[story] = bert_tensor

breakpoint()
# In[25]:


# print('whisper shape:', story2whisper[story].shape)


# In[26]:


# print('fmri shape:', story2fmri[story].shape)


# In[27]:


# print('bert shape:', story2bert[story].shape)


# In[29]:


torch.save(story2fmri_tensor, os.path.join(data_dir, "synth_all_fmri_features.pt"))
torch.save(story2whisper_tensor, os.path.join(data_dir, "synth_all_whisper_features.pt"))
torch.save(story2bert_tensor, os.path.join(data_dir, "synth_all_bert_fmri_features.pt"))
'''

# In[1]:

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


# In[2]:


data_dir = "/nlp/scr/quevedo"
fmri_tensor_file = "synth_all_fmri_features.pt"
whisper_tensor_file = "synth_all_whisper_features.pt"
bert_tensor_file = "synth_all_bert_fmri_features.pt"
whisper_reg_wt_file = "S3_whisper-large_wts_layer32.jbl"


# In[3]:

whisper = torch.load(os.path.join(data_dir, whisper_tensor_file))
bert = torch.load(os.path.join(data_dir, bert_tensor_file))
wt = joblib.load(os.path.join(data_dir, whisper_reg_wt_file))
wt = torch.tensor(wt).float().cuda()


# In[4]:


fmri = torch.load(os.path.join(data_dir, fmri_tensor_file))


# In[3]:


fmri


# In[30]:


whisper_reg_wt_file = "S3_whisper-large_wts_layer32.jbl"


# In[6]:

'''
whisper = story2whisper_tensor
bert = story2bert_tensor
fmri = story2fmri_tensor
wt = joblib.load(os.path.join(data_dir, whisper_reg_wt_file))
wt = torch.tensor(wt).float().cuda()
'''

# In[75]:


p = np.random.permutation(len(fmri))
split_size = int(len(fmri) * 0.90)
train_indices = p[:split_size]
val_indices   = p[split_size:]
train_keys = [list(fmri.keys())[i] for i in train_indices]
val_keys = [list(fmri.keys())[i] for i in val_indices]
train_keys = [k for k in train_keys]
val_keys = [k for k in val_keys]


# In[76]:


# train_fmri    = torch.cat([fmri[k] for k in train_keys])
# val_fmri      = torch.cat([fmri[k] for k in val_keys])
train_bert    = torch.cat([bert[k] for k in train_keys])
# val_bert      = torch.cat([bert[k] for k in val_keys])


# In[7]:


from torch.utils.data import TensorDataset, DataLoader, random_split
from models import LinearDecoder, MLPDecoder


# In[8]:


# In[14]:


decoder = LinearDecoder(512, train_bert.shape[-1]).cuda().float()
# decoder = MLPDecoder(512, train_bert.shape[-1], 2).cuda().float()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(decoder.parameters())


# In[30]:


# In[46]:


top_indices = torch.tensor(np.load("top128ind.npz")['arr_0']).cuda()


# In[23]:

real_features_dir = "features_cnk0.1_ctx16.0"
anth_features_dir = "anth_features_cnk0.1_ctx16.0"
libasr_features_dir = "libasr_features_cnk0.1_ctx16.0"

anth_story_names = [
        ent.replace('.npz', '') for ent in os.listdir(os.path.join(data_dir, anth_features_dir, whisper_model, 'encoder.0')) 
]

lib_story_names = [
        ent.replace('.npz', '') for ent in os.listdir(os.path.join(data_dir, libasr_features_dir, whisper_model, 'encoder.0')) 
]

fmri_train_keys = ['adollshouse', 'adventuresinsayingyes', 'alternateithicatom', 'avatar', 'buck', 'exorcism',
            'eyespy', 'fromboyhoodtofatherhood', 'hangtime', 'haveyoumethimyet', 'howtodraw', 'inamoment',]
synth_train_keys = anth_story_names + lib_story_names

val_keys = ["wheretheressmoke"]


# In[54]:


train_whisper = torch.cat([whisper[k] for k in synth_train_keys if k in whisper])
train_bert_fmri    = torch.cat([bert[k] for k in fmri_train_keys if k in whisper])
train_bert_synth    = torch.cat([bert[k] for k in synth_train_keys if k in whisper])
train_fmri    = torch.cat([fmri[k] for k in fmri_train_keys if k in whisper])
# train_set = torch.cat([train_whisper, train_fmri])


# In[55]:


val_bert      = torch.cat([bert[k] for k in val_keys])
val_fmri      = torch.cat([fmri[k] for k in val_keys])


# In[56]:


fmri_train_dataset = TensorDataset(train_fmri, train_bert_fmri)
synth_train_dataset = TensorDataset(train_whisper, train_bert_synth)
fmri_val_dataset = TensorDataset(val_fmri, val_bert)
# train_dataset = TensorDataset(train_set)


# In[57]:


batch_size = 128
synth_train_loader = DataLoader(synth_train_dataset, batch_size=batch_size, shuffle=True)
fmri_train_loader = DataLoader(fmri_train_dataset, batch_size=batch_size, shuffle=True)
fmri_val_loader = DataLoader(fmri_val_dataset, batch_size=batch_size, shuffle=True)


# In[91]:


decoder = LinearDecoder(512, train_bert_fmri.shape[-1]).cuda().float()
# decoder = MLPDecoder(512, train_bert_fmri.shape[-1], 2).cuda().float()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(decoder.parameters())


# In[92]:


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

# In[93]:


ax.plot(*zip(*real_losses), color="blue")
ax.plot(*zip(*val_losses), color="red")
fig.savefig("loss.png")

# In[94]:


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
# fig.savefig("loss.png")


# In[95]:


# ax.plot(*zip(*losses), color="blue")
# ax.plot(*zip(*val_losses), color="red")
fig.savefig("loss_only_real_14.png")


# In[ ]:




