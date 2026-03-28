import json
import random
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

train_json_path = "./data/forget.json"
val_json_path = "./data/wmdp_test.json"
model_name = "mistralai/Ministral-8B-Instruct-2410"
probe_type = "mlp"
candidate_layers = [28, 32, 36]
max_len_resp = 1024
batch_size = 256
epochs = 10
lr = 1e-4
l1_lambda = 1e-4
eps = 1e-2
alpha = 0.4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token

chat_tmpl = tokenizer.apply_chat_template(
    [
        {"role": "user", "content": "{instruction}"},
        {"role": "assistant", "content": "<SEP>{response}"},
    ],
    tokenize=False,
    add_generation_prompt=False,
).replace(tokenizer.bos_token or "", "")

def load_pairs(path, chat_tmpl):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    pairs = []
    for e in data:
        instruction = e["question"]
        response = e.get("generated_answer", "")
        text = chat_tmpl.format(instruction=instruction, response=response)
        lbl = 1 if e["label"].lower() == "positive" else 0
        pairs.append((text, lbl))
    random.seed(0)
    random.shuffle(pairs)
    return pairs

@torch.inference_mode()
def extract_sequence_for_layer(pairs, layer_id):
    feats, labels = [], []
    for txt, lbl in pairs:
        prompt, resp = txt.split("<SEP>")
        tokenizer.padding_side = "left"
        pt = tokenizer(prompt, return_tensors="pt", padding=False).to(device)
        tokenizer.padding_side = "right"
        rt = tokenizer(
            resp,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_len_resp,
            add_special_tokens=False
        ).to(device)
        inp = torch.cat([pt.input_ids, rt.input_ids], 1)
        mask = torch.cat([pt.attention_mask, rt.attention_mask], 1)
        hs = model(input_ids=inp, attention_mask=mask, output_hidden_states=True).hidden_states
        h = hs[layer_id][0, -max_len_resp:, :].cpu().numpy()
        m = rt.attention_mask.cpu().numpy().squeeze()
        h_seq = h[m == 1].mean(axis=0)
        feats.append(h_seq)
        labels.append(lbl)
    return np.stack(feats), np.array(labels)

def train_probe_once(x_train, y_train, in_dim):
    X = torch.from_numpy(x_train).float().to(device)
    Y = torch.from_numpy(y_train).long().to(device)
    loader = DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True)
    if probe_type == "linear":
        probe = nn.Linear(in_dim, 2).to(device)
    else:
        probe = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        ).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()
    for _ in range(epochs):
        probe.train()
        for xb, yb in loader:
            xb = xb.clone().detach().requires_grad_(True)
            logits_clean = probe(xb)
            loss_clean = ce_loss(logits_clean, yb)
            grad = torch.autograd.grad(
                loss_clean,
                xb,
                retain_graph=True,
                create_graph=False
            )[0]
            adv_x = xb + eps * grad.sign()
            logits_adv = probe(adv_x.detach())
            loss_adv = ce_loss(logits_adv, yb)
            l1_pen = sum(p.abs().sum() for p in probe.parameters())
            loss = loss_clean + alpha * loss_adv + l1_lambda * l1_pen
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return probe

@torch.inference_mode()
def evaluate_on_pairs(probe, pairs, layer_id):
    probe.eval()
    x_seq, y_seq = extract_sequence_for_layer(pairs, layer_id)
    X_seq = torch.from_numpy(x_seq).float().to(device)
    Y_seq = torch.from_numpy(y_seq).long().to(device)
    logits = probe(X_seq)
    preds = logits.argmax(dim=1)
    correct = (preds == Y_seq).sum().item()
    total = Y_seq.size(0)
    return correct / total if total > 0 else 0.0

def find_best_layer(train_pairs, val_pairs, candidate_layers):
    best_layer, best_acc = None, 0.0
    print("Comparing candidate layers...")
    for layer_id in candidate_layers:
        print(f"Testing layer {layer_id} ...")
        x_train, y_train = extract_sequence_for_layer(train_pairs, layer_id)
        probe = train_probe_once(x_train, y_train, x_train.shape[1])
        acc = evaluate_on_pairs(probe, val_pairs, layer_id)
        print(f"Layer {layer_id} validation accuracy = {acc:.4f}")
        if acc > best_acc:
            best_acc, best_layer = acc, layer_id
    print(f"Best layer: {best_layer}, accuracy = {best_acc:.4f}")
    return best_layer, best_acc

if __name__ == "__main__":
    train_pairs = load_pairs(train_json_path, chat_tmpl)
    val_pairs = []
    with open(val_json_path, encoding="utf-8") as f:
        data = json.load(f)
    for e in data:
        resp = e.get("generated_answer", "")
        text = chat_tmpl.format(instruction=e["question"], response=resp)
        lbl = 1 if e["label"].lower() == "positive" else 0
        val_pairs.append((text, lbl))
    best_layer, best_acc = find_best_layer(train_pairs, val_pairs, candidate_layers)
    print(f"Retraining probe on full train set with layer {best_layer} ...")
    x_train_best, y_train_best = extract_sequence_for_layer(train_pairs, best_layer)
    final_probe = train_probe_once(x_train_best, y_train_best, x_train_best.shape[1])
    save_path = "probe_wmdp_Ministral.pt"
    torch.save(final_probe.state_dict(), save_path)
    print(f"Saved probe to {save_path} with layer {best_layer}")