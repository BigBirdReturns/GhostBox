"""Portable supervised fine-tuning run for curated AIP/OTel records.

Executes standard PEFT LoRA, retains base-vs-adapter held-out outcomes, and
packages a self-contained adapter directory. Requires optional torch,
transformers, peft and CUDA-capable runtime. Does not require Foundry.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time


def _load_rows(path):
    rows=[json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    if not rows:
        raise ValueError("Empty training/evaluation split")
    for r in rows:
        m=r.get("messages")
        if not isinstance(m,list) or len(m)!=2 or m[0].get("role")!="user" or m[1].get("role")!="assistant":
            raise ValueError("Expected independently curated user/assistant pairs")
        if not r.get("source_sha256") or not r.get("policy_version") or not r.get("case_id"):
            raise ValueError("Missing provenance/policy/case identity")
    return rows


def run(train_path, eval_path, model_id, out_dir,
        *, max_steps=300, rank=8, batch_size=8, learning_rate=0.0004,
        max_length=384, device="cuda", seed=20261113):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType

    if Path(out_dir).exists():
        raise ValueError("Output exists; do not overwrite accepted model evidence")
    train=_load_rows(train_path)
    evaluate=_load_rows(eval_path)
    if {x["case_id"] for x in train}&{x["case_id"] for x in evaluate}:
        raise ValueError("Train/eval case overlap: contaminated holdout")
    if device=="cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 6))
    start=time.monotonic()
    tok=AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token=tok.eos_token
    dtype=torch.float16 if device=="cuda" else torch.float32
    net=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=dtype)
    net.to(device).eval()

    def prefix(row):
        # Precisely preserve reviewed input. System policy is part of the task
        # interface, not metadata from an uncertain model trace.
        msg=[{"role":"system","content":"Return the requested answer only."},
             {"role":"user","content":row["messages"][0]["content"]}]
        return tok.apply_chat_template(msg,tokenize=True,add_generation_prompt=True)

    def grade(n, rows):
        n.eval()
        correct=0
        preds=[]
        t=time.monotonic()
        for r in rows:
            ids=torch.tensor([prefix(r)],device=device)
            mask=torch.ones_like(ids)
            with torch.no_grad():
                generated=n.generate(input_ids=ids, attention_mask=mask,
                   max_new_tokens=24, do_sample=False,pad_token_id=tok.pad_token_id)
            text=tok.decode(generated[0,ids.shape[-1]:],skip_special_tokens=True).strip()
            desired=r["messages"][1]["content"].strip()
            passed=text==desired
            correct+=int(passed)
            preds.append({"case_id":r["case_id"],"trace_id":r["trace_id"],
                          "policy_version":r["policy_version"],
                          "expected":desired,"predicted":text,"correct":passed})
        return {"correct":correct,"total":len(rows),"accuracy":correct/len(rows),
                "wall_s":round(time.monotonic()-t,3),"predictions":preds}

    before=grade(net,evaluate)
    cfg=LoraConfig(task_type=TaskType.CAUSAL_LM,r=rank,lora_alpha=rank*2,
                  lora_dropout=0.05,target_modules=["q_proj","v_proj"],bias="none")
    net=get_peft_model(net,cfg)
    optim=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                            lr=learning_rate,weight_decay=0.01)

    def training_pair(row):
        first=prefix(row)
        tail=tok.encode(row["messages"][1]["content"]+tok.eos_token,
                        add_special_tokens=False)
        tokens=first+tail
        labels=[-100]*len(first)+tail
        if len(tokens)>max_length:
            # Reject truncation that would destroy supervision context.
            raise ValueError("Training example longer than max_length; raise the explicit limit")
        return torch.tensor(tokens),torch.tensor(labels)

    pairs=[training_pair(r) for r in train]
    from torch.nn.utils.rnn import pad_sequence
    history=[]
    steps=0
    tt=time.monotonic()
    net.train()
    for epoch in range((max_steps*batch_size)//len(pairs)+3):
        order=list(range(len(pairs)))
        random.Random(seed+epoch).shuffle(order)
        for begin in range(0,len(order),batch_size):
            batch=[pairs[i] for i in order[begin:begin+batch_size]]
            X=pad_sequence([p[0] for p in batch],batch_first=True,
                           padding_value=tok.pad_token_id).to(device)
            Y=pad_sequence([p[1] for p in batch],batch_first=True,
                           padding_value=-100).to(device)
            lengths=torch.tensor([len(p[0]) for p in batch],device=device)
            mask=(torch.arange(X.shape[1],device=device).unsqueeze(0)<lengths.unsqueeze(1)).long()
            optim.zero_grad(set_to_none=True)
            loss=net(input_ids=X,attention_mask=mask,labels=Y).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],1.0)
            optim.step()
            history.append(float(loss.item()))
            steps+=1
            if steps%25==0:
                print("step",steps,"mean_loss",round(sum(history[-25:])/25,4),flush=True)
            if steps>=max_steps:
                break
        if steps>=max_steps: break
    training_s=round(time.monotonic()-tt,3)
    after=grade(net,evaluate)
    output=Path(out_dir)
    output.mkdir(parents=True,exist_ok=False)
    adapter_dir=output/"adapter"
    net.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    md={
      "format":"axm-portable-peft-sft/1","source":"independent reviewed OTLP trace curation",
      "model_id":str(model_id),"train_records":len(train),"eval_records":len(evaluate),
      "train_sha256":hashlib.sha256(Path(train_path).read_bytes()).hexdigest(),
      "eval_sha256":hashlib.sha256(Path(eval_path).read_bytes()).hexdigest(),
      "training_steps":steps,"training_wall_s":training_s,
      "loss_first_15":round(sum(history[:15])/len(history[:15]),6),
      "loss_last_15":round(sum(history[-15:])/len(history[-15:]),6),
      "rank":rank,"target_modules":["q_proj","v_proj"],
      "trainable_parameters":sum(p.numel() for p in net.parameters() if p.requires_grad),
      "base_parameters":sum(p.numel() for p in net.parameters()),
      "peak_cuda_allocated_gib":round(torch.cuda.max_memory_allocated()/1024**3,3) if device=="cuda" else None,
      "device":torch.cuda.get_device_name(0) if device=="cuda" else "CPU",
      "eval_before":{k:v for k,v in before.items() if k!="predictions"},
      "eval_after":{k:v for k,v in after.items() if k!="predictions"},
      "wall_s":round(time.monotonic()-start,3),
      "generated_at":datetime.now(timezone.utc).isoformat(),
      "limitations":"No Palantir tenant/benchmark evaluated; PEFT adapter only; base checkpoint dependency retained.",
    }
    (output/"metrics.json").write_text(json.dumps(md,indent=2)+"\n")
    (output/"predictions.json").write_text(json.dumps({"base":before["predictions"],
        "adapter":after["predictions"]},indent=2)+"\n")
    print(json.dumps(md,indent=2),flush=True)
    return md


def main(argv=None):
    p=argparse.ArgumentParser(description="Train portable LoRA from independently curated OTel traces")
    p.add_argument("--train",required=True)
    p.add_argument("--eval",required=True)
    p.add_argument("--model",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--max-steps",type=int,default=300)
    p.add_argument("--rank",type=int,default=8)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    p.add_argument("--max-length",type=int,default=384)
    a=p.parse_args(argv)
    run(a.train,a.eval,a.model,a.out,max_steps=a.max_steps,
        rank=a.rank,batch_size=a.batch_size,device=a.device,max_length=a.max_length)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
