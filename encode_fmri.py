import os
import pickle
import numpy as np
import joblib
from ridge_utils.npp import zs
from ridge_utils.interpdata import lanczosinterp2D
from ridge_utils.util import make_delayed

features_dir = "features_cnk0.1_ctx16.0"
whisper_model = "whisper-tiny"
wordseqs_file = "wordseqs.pkl"
fmri_file = "UTS03_responses.jbl"

if __name__ == "__main__":
    # load wordseqs (transcripts with TR timings)
    with open(wordseqs_file, "rb") as f:
        wordseqs = pickle.load(f)

    # get story names from files
    story_names = [
        ent.replace(".npz", "") for ent in os.listdir(os.path.join(features_dir, "encoder.0"))
    ]

    # mapping from stories to features
    story2features = {}

    for story in story_names:
        if not os.path.isfile(os.path.join(features_dir, f"{story}_snippet_texts.npz")):
            print(f"Skipping {story} because snippet texts were not generated.")
            continue
        snippet_times = np.load(os.path.join(features_dir, f"{story}_times.npz"))['times'][:,1] # shape: (time,)
        snippet_texts = np.load(os.path.join(features_dir, f"{story}_snippet_texts.npz"))['snippet_texts'] # shape: (time,)
        snippet_features = np.load(os.path.join(features_dir, f"{story}.npz"))['features'] # shape: (time, model dim.)

        # now that we have the word-by-word whisper features (snippet_features), we can downsample them to fMRI timings
        downsampled_features, _ = lanczosinterp2D(
            snippet_features, snippet_times, wordseqs[story].tr_times)

        story2features[story] = downsampled_features
    
    # load fMRI data
    story2fmri = joblib.load("UTS03_responses.jbl") # Located in story_responses folder

    # split into train and test
    train_stories = story_names[:-1]
    test_stories = story_names[-1:]

    # don't mind the magic offsets here
    train_feats = np.nan_to_num(np.vstack([zs(story2features[story][10:-5]) for story in train_stories]))
    train_fmri  = np.vstack([story2fmri[story] for story in train_stories])
    test_feats  = np.nan_to_num(np.vstack([zs(story2features[story][50:-5]) for story in test_stories]))
    test_fmri   = np.vstack([story2fmri[story][40:] for story in test_stories])

    # add FIR delays
    delayed_train_feats = make_delayed(train_feats, 4)
    delayed_test_feats  = make_delayed(test_feats,  4)

    # TODO train regression model OR load pre-trained regression model from Box
    # TODO evaluate model on test data


