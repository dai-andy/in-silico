import os
import pickle
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm

data_dir = "/nlp/scr/quevedo/"
story2snippets_file = "training_story2snippets.pkl"
story2bert_file = "training_story2bert.pkl"

def load_bert():
    print("loading BERT...")
    bert_model = "bert-base-uncased"
    bert = AutoModel.from_pretrained(bert_model).cuda()
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model)
    bert_tokenizer.padding_side = "left"
    print("done")
    return bert, bert_tokenizer

def get_bert_embedding(seq, tok, model, layer_num=-1):
    embeddings = []
    toks = tok(seq, padding=True)
    input_ids = torch.tensor(toks['input_ids'])
    attention_mask = torch.tensor(toks['attention_mask'])
    input_ids = input_ids[:, -tok.model_max_length:]
    attention_mask = attention_mask[:, -tok.model_max_length:]
    batch_size = 256
    for ids, mask in zip(torch.split(input_ids, batch_size, dim=0), torch.split(attention_mask, batch_size, dim=0)):
        ids = ids.cuda()
        mask = mask.cuda()
        with torch.no_grad():
            h = model(
                input_ids=ids,
                attention_mask=mask,
                output_hidden_states=True).hidden_states
        layer_h = h[layer_num]
        cls_h = layer_h[ids == tok.cls_token_id]
        embeddings.append(cls_h.cpu())
    return torch.cat(embeddings)

if __name__ == "__main__":
    bert, tok = load_bert()

    with open(os.path.join(data_dir, story2snippets_file), "rb") as f:
        story2snippets = pickle.load(f)

    story2bert = {}
    for story in tqdm(story2snippets):
        bert_features = get_bert_embedding(list(story2snippets[story]), tok, bert)
        story2bert[story] = bert_features

    with open(os.path.join(data_dir, story2bert_file), "wb") as f:
        pickle.dump(story2bert, f)




