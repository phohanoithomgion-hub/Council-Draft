import os
import time
import queue
import threading
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer

os.environ["HF_HUB_DISABLE_HF_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.pop("HF_DATASETS_OFFLINE", None)

def _load_dataset_with_retry(repo_id, config, split="train", max_retries=6):
    for attempt in range(max_retries):
        try:
            if config is None:
                return load_dataset(repo_id, split=split, streaming=True)
            return load_dataset(repo_id, config, split=split, streaming=True)
        except Exception as e:
            wait = min(2 ** attempt, 30)
            print(f"[dataset] load {repo_id} failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait}s")
            time.sleep(wait)
    return None

class StreamingSFTDataset(IterableDataset):
    def __init__(self, tokenizer, max_seq_len=512, max_tokens=50_000_000_000):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_tokens = max_tokens
        self.datasets = [
            ("HuggingFaceTB/finemath", "finemath-4plus"),
            ("meituan-longcat/R-HORIZON-training-data", None),
            ("open-thoughts/OpenThoughts-114k", None),
            ("AI-MO/NuminaMath-CoT", None),
            ("nvidia/OpenMathReasoning", None),
            ("open-r1/OpenR1-Math-220k", None),
            ("openai/gsm8k", "main"),
            ("DigitalLearningGmbH/MATH-lighteval", None),
            ("TIGER-Lab/MathInstruct", None),
            ("meta-math/MetaMathQA", None),
        ]
        self.prompt_template = "Problem: {prompt}\nAnswer: "

    def _worker(self, q):
        try:
            total_tokens = 0
            for repo_id, config in self.datasets:
                ds = _load_dataset_with_retry(repo_id, config, split="train")
                if ds is None:
                    print(f"[dataset] skipping {repo_id} after all retries")
                    continue

                for example in ds:
                    if total_tokens >= self.max_tokens:
                        q.put(None)
                        return
                    
                    item = self._extract_qa(example)
                    if item is None:
                        continue
                    
                    prompt_raw, answer_raw = item
                    if len(answer_raw.strip()) < 10:
                        continue
                    
                    prompt_text = self.prompt_template.format(prompt=prompt_raw.strip())
                    answer_text = answer_raw.strip() + self.tokenizer.eos_token
                    
                    prompt_ids_full = self.tokenizer(prompt_text, truncation=False, add_special_tokens=False)["input_ids"]
                    prompt_len = len(prompt_ids_full)

                    if prompt_len >= self.max_seq_len - 10:
                        continue

                    enc_prompt = self.tokenizer(prompt_text, truncation=True, max_length=256, padding="max_length", return_tensors="pt")
                    prompt_ids = enc_prompt["input_ids"].squeeze(0)
                    prompt_mask = enc_prompt["attention_mask"].squeeze(0)

                    full_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"] + \
                               self.tokenizer(answer_text, add_special_tokens=False)["input_ids"]
                    if len(full_ids) > self.max_seq_len:
                        full_ids = full_ids[:self.max_seq_len]
                    pad_len = self.max_seq_len - len(full_ids)
                    input_ids = torch.tensor(full_ids + [self.tokenizer.pad_token_id] * pad_len, dtype=torch.long)
                    attention_mask = torch.tensor([1] * len(full_ids) + [0] * pad_len, dtype=torch.long)
                    
                    labels = input_ids.clone()
                    labels[attention_mask == 0] = -100
                    labels[:prompt_len] = -100
                    
                    total_tokens += (labels != -100).sum().item()

                    q.put({
                        "input_ids": input_ids,
                        "labels": labels,
                        "prompt_ids": prompt_ids,
                        "prompt_mask": prompt_mask,
                    })
            q.put(None)
        except Exception as e:
            q.put(e)

    def __iter__(self):
        q = queue.Queue(maxsize=32)
        t = threading.Thread(target=self._worker, args=(q,), daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    def _extract_qa(self, ex):
        if "conversations" in ex:
            user_msg = next((c["content"] for c in ex["conversations"] if c.get("role") == "user"), None)
            assistant_msg = next((c["content"] for c in ex["conversations"] if c.get("role") == "assistant"), None)
            if user_msg and assistant_msg:
                return user_msg, assistant_msg
        for k1, k2 in [("problem", "solution"), ("question", "answer"), ("instruction", "output"), ("input", "output")]:
            if k1 in ex and k2 in ex:
                return str(ex[k1]), str(ex[k2])
        if "text" in ex:
            text = ex["text"]
            for marker in ["\nAnswer:", "\nSolution:", "\nOutput:"]:
                if marker in text:
                    parts = text.split(marker, 1)
                    return parts[0].strip(), parts[1].strip()
        return None


class StreamingRLDataset(IterableDataset):
    def __init__(self, tokenizer, max_prompt_len=256, max_tokens=10_000_000_000):
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        self.max_tokens = max_tokens
        self.datasets = [
            ("meituan-longcat/R-HORIZON-training-data", None),
            ("openai/gsm8k", "main"),
            ("AI-MO/NuminaMath-CoT", None),
            ("nvidia/OpenMathReasoning", None),
            ("open-thoughts/OpenThoughts-114k", None),
        ]
        self.prompt_template = (
            "Example:\n"
            "Problem: What is 2 + 2?\n"
            "Thinking:\n"
            "I need to add 2 and 2. The answer is 4.\n"
            "Answer:\n"
            "4\n\n"
            "Problem: {prompt}\n"
            "Thinking:\n"
        )

    def __iter__(self):
        total = 0
        for repo_id, config in self.datasets:
            ds = _load_dataset_with_retry(repo_id, config, split="train")
            if ds is None:
                print(f"[dataset] skipping {repo_id} after all retries")
                continue

            for ex in ds:
                if total >= self.max_tokens:
                    return
                item = self._extract(ex)
                if item is None:
                    continue
                prompt, gt = item
                prompt_text = self.prompt_template.format(prompt=prompt.strip())
                enc = self.tokenizer(
                    prompt_text, truncation=True, max_length=self.max_prompt_len,
                    padding="max_length", return_tensors="pt"
                )
                prompt_ids = enc["input_ids"].squeeze(0)
                prompt_mask = enc["attention_mask"].squeeze(0)
                total += prompt_ids.numel()
                yield {"prompt_ids": prompt_ids, "prompt_mask": prompt_mask, "ground_truth": gt}

    def _extract(self, ex):
        if "question" in ex and "answer" in ex:
            q = ex["question"]
            a = ex["answer"].split("####")[-1].strip() if "####" in ex["answer"] else ex["answer"]
            return q, a
        if "problem" in ex and "solution" in ex:
            return ex["problem"], ex["solution"][-200:]
        if "instruction" in ex and "output" in ex:
            return ex["instruction"], ex["output"]
        return None


class StreamingPretrainDataset(IterableDataset):
    DATASET_SPECS = {
        "xerv_pure": {
            "repo_id": "Xerv-AI/PURE",
            "config": "default",
            "weight": 0.01,
        },
        "automathtext-v2-ultra": {
            "repo_id": "OpenSQZ/AutoMathText-V2",
            "config": "automathtext-v2-ultra",
            "weight": 0.99,
        },
    }

    def __init__(self, tokenizer, max_seq_len=512, config_names=None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        if config_names is None:
            config_names = ["xerv_pure", "automathtext-v2-ultra"]
        if isinstance(config_names, str):
            config_names = [config_names]
        self.config_names = config_names

    def _process_example(self, example):
        q = example.get("question") or example.get("problem") or ""
        r = example.get("reasoning") or example.get("solution") or ""
        a = example.get("answer") or ""
        raw_text = example.get("text") or ""

        if q and a:
            text = f"Problem: {q}\nThinking:\n{r}\nAnswer:\n{a}"
        elif raw_text:
            text = raw_text
        else:
            return None

        if len(text) < 50:
            return None

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "n_tokens": (labels != -100).sum().item(),
        }

    def __iter__(self):
        from datasets import interleave_datasets

        streams = []
        weights = []
        for name in self.config_names:
            if name not in self.DATASET_SPECS:
                print(f"[PretrainDataset] unknown config '{name}', skip")
                continue
            spec = self.DATASET_SPECS[name]
            print(f"[PretrainDataset] loading {spec['repo_id']} (config={spec['config']})")
            ds = _load_dataset_with_retry(
                spec["repo_id"], spec["config"], split="train"
            )
            if ds is None:
                print(f"[PretrainDataset] skip '{name}' after retries")
                continue
            streams.append(ds)
            weights.append(spec["weight"])

        if not streams:
            print("[PretrainDataset] no stream available, exit")
            return

        total_w = sum(weights)
        probs = [w / total_w for w in weights]

        if len(streams) == 1:
            mixed = streams[0]
        else:
            mixed = interleave_datasets(
                streams,
                probabilities=probs,
                seed=42,
                stopping_strategy="first_exhausted",
            )

        for example in mixed:
            out = self._process_example(example)
            if out is not None:
                yield out

    def _load_fallback(self):
        """Fallback nếu Xerv-AI/PURE không load được (gated, offline, etc.)"""
        fallbacks = [
            ("nvidia/OpenMathReasoning", "cot"),
            ("open-thoughts/OpenThoughts3-1.2M", None),
            ("open-thoughts/OpenThoughts2-1M", None),
            ("open-r1/OpenR1-Math-220k", "default"),
            ("AI-MO/NuminaMath-CoT", None),
            ("HuggingFaceTB/finemath", "finemath-4plus"),
            ("HuggingFaceTB/finemath", "finemath-3plus"),
        ]
        for repo_id, config in fallbacks:
            print(f"[fallback] Trying {repo_id} (config={config})...")
            ds = _load_dataset_with_retry(repo_id, config, split="train")
            if ds is not None:
                print(f"[fallback] OK: {repo_id}")
                return ds
        return None

class PackedPretrainDataset(IterableDataset):
    def __init__(self, tokenizer, max_seq_len=16000, max_tokens_per_batch=16000, config_names=None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_tokens_per_batch = max_tokens_per_batch
        if config_names is None:
            config_names = ["default"]
        if isinstance(config_names, str):
            config_names = [config_names]
        self.config_names = config_names
        self.base = StreamingPretrainDataset(
            tokenizer, max_seq_len=max_seq_len, config_names=config_names
        )

    def __iter__(self):
        buffer_ids = []
        buffer_lens = []
        current = 0

        for sample in self.base:
            ids = sample["input_ids"][sample["attention_mask"] == 1]
            if ids.numel() == 0:
                continue
            if ids.numel() > self.max_seq_len:
                ids = ids[:self.max_seq_len]
            if ids.numel() > self.max_tokens_per_batch:
                ids = ids[:self.max_tokens_per_batch]

            if current + ids.numel() > self.max_tokens_per_batch and buffer_ids:
                yield self._pack(buffer_ids, buffer_lens)
                buffer_ids = []
                buffer_lens = []
                current = 0

            buffer_ids.append(ids)
            buffer_lens.append(ids.numel())
            current += ids.numel()

        if buffer_ids:
            yield self._pack(buffer_ids, buffer_lens)

    def _pack(self, ids_list, lens_list):
        real_tokens = sum(lens_list)
        packed = torch.cat(ids_list, dim=0)
        pad_len = self.max_tokens_per_batch - real_tokens
        
        if pad_len > 0:
            pad = torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)
            packed = torch.cat([packed, pad], dim=0)
            lens_list.append(pad_len)
            
        labels = torch.cat([torch.cat(ids_list, dim=0), torch.full((max(0, pad_len),), -100, dtype=torch.long)])
        
        cu = torch.tensor(
            [0] + list(torch.tensor(lens_list).cumsum(0).tolist()),
            dtype=torch.int32,
        )
        return {
            "input_ids": packed,
            "labels": labels,
            "cu_seqlens": cu,
            "n_tokens": real_tokens,
        }


def get_tokenizer(model_name="gpt2"):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.model_max_length = 10_000_000
    return tok