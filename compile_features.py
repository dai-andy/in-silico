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

with open(os.path.join(data_dir, wordseqs_file), "rb") as f:
    wordseqs = pickle.load(f)
fmri = joblib.load(os.path.join(data_dir, fmri_file))


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

# with open(os.path.join(data_dir, story2fmri_file), "wb") as f:
#    pickle.dump(story2fmri, f)
# print('dumped fmri data')
#
# with open(os.path.join(data_dir, story2whisper_file), "wb") as f:
#    pickle.dump(story2whisper, f)
# print('dumped whisper features')
#
# with open(os.path.join(data_dir, story2snippets_file), "wb") as f:
#    pickle.dump(story2snippets, f)
# print('dumped snippets')

# convert features into pytorch tensors
story2whisper_tensor = {}
story2fmri_tensor = {}
story2bert_tensor = {}

story2bert_file = "synth_training_story2bert.pkl"
with open(os.path.join(data_dir, story2bert_file), "rb") as f:
    story2bert = pickle.load(f)

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

torch.save(story2fmri_tensor, os.path.join(data_dir, "synth_all_fmri_features.pt"))
torch.save(story2whisper_tensor, os.path.join(data_dir, "synth_all_whisper_features.pt"))
torch.save(story2bert_tensor, os.path.join(data_dir, "synth_all_bert_fmri_features.pt"))
