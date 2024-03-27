import os
import pickle
import numpy as np
import torch
import joblib
from ridge_utils.npp import zs
from ridge_utils.interpdata import lanczosinterp2D
from ridge_utils.util import make_delayed, spe_and_cc_norm

features_dir = "/nlp/scr/quevedo/features_cnk0.1_ctx16.0/whisper-large"
whisper_model = "whisper-large"
wordseqs_file = "/nlp/scr/quevedo/encoding-model-scaling-laws/wordseqs.pkl"
repeat_sessions_file = "/nlp/scr/quevedo/encoding-model-scaling-laws/tensessions_wheretheressmoke_S03.jbl"
fmri_file = "/nlp/scr/quevedo/encoding-model-scaling-laws/tensessions_wheretheressmoke_S03.jbl"

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
        if story not in wordseqs:
            print(f"Skipping {story} because it wasn't found in wordseqs.")
            continue
        snippet_times = np.load(os.path.join(features_dir, f"{story}_times.npz"))['times'][:,1] # shape: (time,)
        snippet_features = np.load(os.path.join(features_dir, f"{story}.npz"))['features'] # shape: (time, model dim.)
        # now that we have the word-by-word whisper features (snippet_features), we can downsample them to fMRI timings
        downsampled_features, _ = lanczosinterp2D(
            snippet_features, snippet_times, wordseqs[story].tr_times)

        story2features[story] = downsampled_features

    test_stories = ["wheretheressmoke"]
    test_feats  = np.nan_to_num(np.vstack([zs(story2features[story][50:-5]) for story in test_stories]))

    # add FIR delays
    delayed_test_feats = make_delayed(test_feats,  range(1, 5))

    wt = joblib.load("/nlp/scr/quevedo/S3_whisper-large_wts_layer32.jbl")

    pred_test = np.dot(delayed_test_feats, wt)
    torch.save(pred_test, 'predicted_wheretheressmoke.pth')

    # load the multi-trial fMRI data for a single story
    orig_test = joblib.load("/nlp/scr/quevedo/encoding-model-scaling-laws/tensessions_wheretheressmoke_S03.jbl")

    # how do the predictions compare to the ground truth?
    SPE, cc_norm, cc_max, corrs_unnorm = spe_and_cc_norm(orig_test[:, 40:, :], pred_test, max_flooring=0.25)
    print(f"max correlation: {corrs_unnorm.max()}, mean correlation: {corrs_unnorm.mean()}")

    torch.save(SPE, 'SPE_wheretheressmoke.pth')
    torch.save(cc_norm, 'cc_norm_wheretheressmoke.pth')
    torch.save(cc_max, 'cc_max.pth')
    torch.save(corrs_unnorm, 'corrs_unnorm.pth')

    # if we wanted to train our own regression, we could load the fMRI data:

    # story2fmri = joblib.load("/nlp/scr/quevedo/encoding-model-scaling-laws/UTS03_responses.jbl") # Located in story_responses folder
    # train_feats = np.nan_to_num(np.vstack([zs(story2features[story][10:-5]) for story in train_stories]))
    # train_fmri  = np.vstack([story2fmri[story] for story in train_stories])

