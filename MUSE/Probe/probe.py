import json
import random
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "muse-bench/MUSE-Books_target"
FORGET_FILE = "../data/books/raw/forget.json"
RETAIN_FILE = "../data/books/raw/retain1.json"
CANDIDATE_LAYERS = [32]
MAX_LEN = 2048
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
EPS = 1e-2
ALPHA = 0.4
L1_LAMBDA = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, output_hidden_states=True).to(DEVICE)
model.eval()

def load_texts(path, label):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = data if isinstance(data, list) else [json.loads(line) for line in f]
    return [(txt, label) for txt in texts]

forget_pairs = load_texts(FORGET_FILE, 1)
retain_pairs = load_texts(RETAIN_FILE, 0)
pairs = forget_pairs + retain_pairs
random.shuffle(pairs)

@torch.inference_mode()
def extract_sequence_for_layer(pairs, layer_id):
    feats, labels = [], []
    for text, lbl in pairs:
        toks = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN
        ).to(DEVICE)
        outputs = model(
            input_ids=toks.input_ids,
            attention_mask=toks.attention_mask,
            output_hidden_states=True
        )
        hidden = outputs.hidden_states[layer_id]
        mask = toks.attention_mask.cpu().numpy().squeeze()
        h = hidden[0].cpu().numpy()
        pooled = h[mask == 1].mean(axis=0)
        feats.append(pooled)
        labels.append(lbl)
    return np.stack(feats), np.array(labels)

def train_probe_once(x_train, y_train, in_dim, probe_type="mlp"):
    X = torch.from_numpy(x_train).float().to(DEVICE)
    Y = torch.from_numpy(y_train).long().to(DEVICE)
    loader = DataLoader(TensorDataset(X, Y), batch_size=BATCH_SIZE, shuffle=True)
    if probe_type == "linear":
        probe = nn.Linear(in_dim, 2).to(DEVICE)
    else:
        probe = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2)
        ).to(DEVICE)
    optimizer = torch.optim.Adam(probe.parameters(), lr=LR)
    ce_loss = nn.CrossEntropyLoss()
    for ep in range(1, EPOCHS + 1):
        probe.train()
        total, correct, cumsum = 0, 0, 0.0
        for xb, yb in loader:
            xb = xb.clone().detach().requires_grad_(True)
            logits_clean = probe(xb)
            loss_clean = ce_loss(logits_clean, yb)
            grad = torch.autograd.grad(
                loss_clean, xb,
                retain_graph=True, create_graph=False
            )[0]
            adv_x = xb + EPS * grad.sign()
            logits_adv = probe(adv_x.detach())
            loss_adv = ce_loss(logits_adv, yb)
            l1_pen = sum(p.abs().sum() for p in probe.parameters())
            loss = loss_clean + ALPHA * loss_adv + L1_LAMBDA * l1_pen
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += yb.size(0)
            correct += (logits_clean.argmax(1) == yb).sum().item()
            cumsum += loss_clean.item() * yb.size(0)
        print(f"Epoch {ep}/{EPOCHS}  CE {cumsum/total:.4f}  acc {correct/total:.4f}")
    return probe

@torch.inference_mode()
def evaluate_probe(probe, pairs, layer_id):
    x_val, y_val = extract_sequence_for_layer(pairs, layer_id)
    Xv = torch.from_numpy(x_val).float().to(DEVICE)
    Yv = torch.from_numpy(y_val).long().to(DEVICE)
    logits = probe(Xv)
    preds = logits.argmax(dim=1)
    return (preds == Yv).float().mean().item()

def find_best_layer(train_pairs, val_pairs):
    best_layer, best_acc = None, 0
    for layer in CANDIDATE_LAYERS:
        x_tr, y_tr = extract_sequence_for_layer(train_pairs, layer)
        probe = train_probe_once(x_tr, y_tr, x_tr.shape[1])
        acc = evaluate_probe(probe, val_pairs, layer)
        print(f"Layer {layer} acc={acc:.4f}")
        if acc > best_acc:
            best_acc, best_layer = acc, layer
    print(f"Best layer: {best_layer}, acc={best_acc:.4f}")
    return best_layer

if __name__ == '__main__':
    split = int(0.8 * len(pairs))
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]
    best_layer = find_best_layer(train_pairs, val_pairs)
    x_tr, y_tr = extract_sequence_for_layer(train_pairs, best_layer)
    final_probe = train_probe_once(x_tr, y_tr, x_tr.shape[1])
    torch.save(final_probe.state_dict(), 'final_probe.pt')
    print("Saved final probe.")