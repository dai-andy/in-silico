import os
import pickle
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from models import LinearDecoder, MLPDecoder
from tqdm import tqdm
import matplotlib.pyplot as plt

data_dir = "/nlp/scr/quevedo"
whisper_tensor_file = "all_whisper_features.pt"
bert_tensor_file = "all_bert_features.pt"

whisper = torch.load(os.path.join(data_dir, whisper_tensor_file))
bert = torch.load(os.path.join(data_dir, bert_tensor_file))

p = np.random.permutation(len(whisper))
split_size = int(len(whisper) * 0.8)
train_indices = p[:split_size]
val_indices   = p[split_size:]

train_whisper = torch.cat([whisper[i] for i in train_indices])
train_bert    = torch.cat([bert[i] for i in train_indices])
val_whisper   = torch.cat([whisper[i] for i in val_indices])
val_bert      = torch.cat([bert[i] for i in val_indices])

train_whisper = train_whisper
train_bert = train_bert

train_dataset = TensorDataset(train_whisper, train_bert)
val_dataset = TensorDataset(val_whisper, val_bert)
print("train dataset size:", len(train_dataset))
print("val dataset size:", len(val_dataset))

batch_size = 512
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

decoder = LinearDecoder(train_whisper.shape[-1], train_bert.shape[-1]).cuda().float()
# decoder = MLPDecoder(train_whisper.shape[-1], train_bert.shape[-1], 2).cuda().float()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(decoder.parameters())

fig, ax = plt.subplots()
losses = []
val_losses = []
num_epochs = 40
for epoch in tqdm(range(num_epochs)):
    for i, (x, y) in enumerate(train_loader):
        batch_num = epoch * len(train_loader) + i
        decoder.train()
        optimizer.zero_grad()

        x = x.cuda().float()
        y = y.cuda().float()
        y_pred = decoder(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        losses.append((batch_num, loss.item()))

        optimizer.step()

        if i % 8 == 0:
            decoder.eval()
            batch_val_losses = []
            for x, y in val_loader:
                x = x.cuda().float()
                y = y.cuda().float()
                y_pred = decoder(x)
                val_loss = loss_fn(y_pred, y)
                batch_val_losses.append(val_loss.item())
            val_losses.append((batch_num, torch.tensor(batch_val_losses).mean().item()))
    print(f"epoch {epoch} val loss: {val_loss.item()}")

ax.plot(*zip(*losses), color="blue")
ax.plot(*zip(*val_losses), color="red")
fig.savefig("loss.png")

