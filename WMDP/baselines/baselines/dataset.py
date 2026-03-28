from .utils import read_text, pad_or_trim_tensor

from typing import List, Tuple
from pathlib import Path
import json

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import AutoTokenizer


class DefaultDataset(Dataset):

    def __init__(
        self,
        file_path: str,
        tokenizer: AutoTokenizer | None = None,
        max_len: int | None = 4096,
        add_bos_token: bool = True
    ):
        if Path(file_path).suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
            if isinstance(data[0], str):
                self.strings = data
            elif isinstance(data[0], dict) and 'text' in data[0] \
                    and isinstance(data[0]['text'], str):
                self.strings = [d['text'] for d in data]
                if 'input_ids' in data[0]:
                    self.input_ids = [torch.tensor(d['input_ids']) for d in data]
                    return; # Done, since we have `input_ids` ready.
            else:
                raise ValueError("Format of this `.json` file is not recognized.")

            assert tokenizer is not None, "Tokenizer must be specified."

            self.input_ids = []
            for s in self.strings:
                encoding: torch.Tensor = tokenizer(
                    s,
                    add_special_tokens=add_bos_token,
                    return_tensors='pt'
                ).input_ids[0]
                encoding = pad_or_trim_tensor(
                    encoding,
                    target_length=max_len,
                    padding_value=tokenizer.pad_token_id
                )
                self.input_ids.append(encoding)

            return; # end if Path(file_path).suffix == '.json'

        assert Path(file_path).suffix == '.txt'

        tokens = tokenizer(read_text(file_path), add_special_tokens=False, return_tensors='pt').input_ids[0]
        assert len(tokens.shape) == 1, "Debug error: Tokens not 1-dimensional"

        if add_bos_token:
            self.input_ids = [
                F.pad(
                    tokens[i : i + max_len - 1], (1, 0),
                    value=tokenizer.bos_token_id
                )
                for i in range(0, len(tokens), max_len - 1)
            ]
        else:
            self.input_ids = [
                tokens[i : i + max_len]
                for i in range(0, len(tokens), max_len)
            ]

        # Rotate the tokens if the last `input_ids` isn't filled to max_len
        if len(self.input_ids[-1]) < max_len:
            self.input_ids[-1] = torch.concat(
                [self.input_ids[-1], self.input_ids[0]], dim=-1
            )[:max_len]

        # Original strings
        self.strings = tokenizer.batch_decode(self.input_ids, skip_special_tokens=True)

        json_output_path = Path(file_path).with_suffix('.json')
        print(f"Saving split text chunks from '{file_path}' to '{json_output_path}'...")
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(self.strings, f, ensure_ascii=False, indent=4)
        print("Save complete.")

        pass    # def __init__()


    def __getitem__(self, index):
        return self.input_ids[index]


    def __len__(self):
        return len(self.input_ids)


    def get_collate_fn(self):

        def collate_fn(batch: List[torch.Tensor]):
            batch = torch.stack(batch)
            return {
                "input_ids": batch,
                "labels": batch.clone()
            }

        return collate_fn



# wmdp
SYSTEM_PROMPT_FORGET = "You are a helpful assistant. The following are questions (with answers) about biology or something I want you to answer."
SYSTEM_PROMPT_RETAIN = "You are a helpful assistant."

class ForgetRetainDataset(Dataset):
    def __init__(
        self,
        forget_file_path: str,
        tokenizer,
        retain_file_path: str | None = None,
        max_len: int = 1024,
        pad: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad = pad

        self.forget_samples = []
        for line in Path(forget_file_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            msgs = data.get("messages", [])
            if len(msgs) < 2:
                continue
            self.forget_samples.append({
                "user": msgs[0]["content"].strip(),
                "assistant": msgs[1]["content"].strip()
            })
        if not self.forget_samples:
            raise RuntimeError(f"No forget samples loaded from {forget_file_path}")

        if retain_file_path:
            self.retain_samples = []
            for line in Path(retain_file_path).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                data = json.loads(line)
                msgs = data.get("messages", [])
                if len(msgs) < 2:
                    continue
                self.retain_samples.append({
                    "user": msgs[0]["content"].strip(),
                    "assistant": msgs[1]["content"].strip()
                })
            if not self.retain_samples:
                raise RuntimeError(f"No retain samples loaded from {retain_file_path}")
        else:
            self.retain_samples = self.forget_samples

        self.length = len(self.forget_samples)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        IGN = -100

        def encode_pair(sample, system_prompt):
            prompt_text = f"{system_prompt}\n{sample['user']}"
            full_text = f"{prompt_text}\n{sample['assistant']}"

            enc_full = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_len,
                return_attention_mask=True
            )
            enc_prompt = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.max_len,
                return_attention_mask=False
            )

            input_ids = torch.tensor(enc_full["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(enc_full["attention_mask"], dtype=torch.float)
            prompt_len = len(enc_prompt["input_ids"])

            L = input_ids.size(0)
            if self.pad:
                if L < self.max_len:
                    pad_len = self.max_len - L
                    input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)], dim=0)
                    attention_mask = torch.cat([attention_mask, torch.zeros(pad_len)], dim=0)
                else:
                    input_ids = input_ids[: self.max_len]
                    attention_mask = attention_mask[: self.max_len]

            labels = input_ids.clone()
            labels[:prompt_len] = IGN

            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        fs = self.forget_samples[idx]
        rs = self.retain_samples[idx % len(self.retain_samples)]
        x_f = encode_pair(fs, SYSTEM_PROMPT_FORGET)
        x_r = encode_pair(rs, SYSTEM_PROMPT_RETAIN)
        return x_f, x_r

    def get_collate_fn(self):
        def collate_fn(batch):
            batch_f = [item[0] for item in batch]
            batch_r = [item[1] for item in batch]

            dict_forget = {
                key: torch.stack([sample[key] for sample in batch_f])
                for key in ["input_ids", "attention_mask", "labels"]
            }
            dict_retain = {
                key: torch.stack([sample[key] for sample in batch_r])
                for key in ["input_ids", "attention_mask", "labels"]
            }
            return dict_forget, dict_retain

        return collate_fn
