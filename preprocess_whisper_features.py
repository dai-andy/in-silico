import os
import numpy as np
from ridge_utils.npp import zs
from ridge_utils.interpdata import lanczosinterp2D, simulate
from ridge_utils.util import make_delayed
import pickle
import torch
from tqdm import tqdm

data_dir = "/nlp/scr/quevedo/"
features_dir = "features_cnk0.1_ctx16.0"
whisper_model = "whisper-large"
wordseqs_file = "wordseqs.pkl"
story2whisper_file = "training_story2whisper.pkl"
story2snippets_file = "training_story2snippets.pkl"

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

if __name__ == "__main__":
    # get story names from files
    story_names = [
        ent.replace(".npz", "") for ent in os.listdir(os.path.join(data_dir, features_dir, whisper_model, "encoder.0"))
    ]

    # mapping from stories to features
    story2whisper = {}
    # mapping from stories to sequences of text snippets (transcriptions)
    story2snippets = {}
    for story in tqdm(story_names):
        if not os.path.isfile(os.path.join(data_dir, features_dir, whisper_model, f"{story}_snippet_texts.npz")):
            print(f"Skipping {story} because snippet texts were not generated.")
            continue
        snippet_times = np.load(os.path.join(data_dir, features_dir, whisper_model, f"{story}_times.npz"))['times'][:,1] # shape: (time,)
        snippet_texts = np.load(os.path.join(data_dir, features_dir, whisper_model, f"{story}_snippet_texts.npz"))['snippet_texts'] # shape: (time,)

        snippet_features = np.load(os.path.join(data_dir, features_dir, whisper_model, f"{story}.npz"))['features'] # shape: (time, model dim.)
        # now that we have the word-by-word whisper features (snippet_features), we can downsample them to fMRI timings
        downsampled_features, sincmat = lanczosinterp2D(
            snippet_features, snippet_times, simulate(snippet_times[-1] // 2, offset=-9))
        trimmed_features = downsampled_features[50:-5]
        normalized_features = np.nan_to_num(zs(trimmed_features))
        delayed_features = make_delayed(normalized_features, range(1, 5))
        downsampled_snippets = get_interpolated_snippets(downsampled_features, sincmat, snippet_texts)
        trimmed_snippets = downsampled_snippets[50:-5]

        assert delayed_features.shape[0] == trimmed_snippets.shape[0]

        story2whisper[story] = delayed_features
        story2snippets[story] = trimmed_snippets

    
    with open(os.path.join(data_dir, story2whisper_file), "wb") as f:
        pickle.dump(story2whisper, f)
    with open(os.path.join(data_dir, story2snippets_file), "wb") as f:
        pickle.dump(story2snippets, f)
    breakpoint()



