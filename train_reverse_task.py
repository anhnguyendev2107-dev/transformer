"""
Reverse task: encoder đọc chuỗi số ngẫu nhiên, decoder sinh lại theo thứ tự ngược.

Khó hơn copy task: decoder phải attend ngược lại encoder (vị trí cuối → đầu),
buộc cross-attention học được pattern reversal thay vì alignment đối xứng.

Chạy: python train_reverse_task.py
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.optim as optim

from transformer import Transformer

PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2
NUM_DIGITS = 12  # số token "thực" (vocab tổng = NUM_DIGITS + 3 cho pad/sos/eos)
VOCAB_SIZE = NUM_DIGITS + 3
MIN_LEN, MAX_LEN = 4, 10

D_MODEL = 128
NUM_HEADS = 8
ENC_LAYERS = DEC_LAYERS = 2
D_FF = 512
BATCH = 32
STEPS = 3000
LR = 3e-4
EVAL_EVERY = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_batch(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sinh ngẫu nhiên `batch_size` chuỗi, target = reverse của src."""
    seqs = [
        [random.randint(3, NUM_DIGITS + 2) for _ in range(random.randint(MIN_LEN, MAX_LEN))]
        for _ in range(batch_size)
    ]
    L = max(len(s) for s in seqs)

    src = torch.full((batch_size, L), PAD_IDX, dtype=torch.long, device=device)
    tgt_in = torch.full((batch_size, L + 1), PAD_IDX, dtype=torch.long, device=device)
    tgt_out = torch.full((batch_size, L + 1), PAD_IDX, dtype=torch.long, device=device)

    for i, s in enumerate(seqs):
        rev = list(reversed(s))
        ln = len(s)
        src[i, :ln] = torch.tensor(s, device=device)
        tgt_in[i, 0] = SOS_IDX
        tgt_in[i, 1 : ln + 1] = torch.tensor(rev, device=device)
        tgt_out[i, :ln] = torch.tensor(rev, device=device)
        tgt_out[i, ln] = EOS_IDX
    return src, tgt_in, tgt_out


@torch.no_grad()
def greedy_decode(model: Transformer, src: torch.Tensor, max_len: int) -> torch.Tensor:
    """Sinh autoregressive: bắt đầu từ <sos>, mỗi bước lấy argmax, dừng khi gặp <eos>."""
    model.eval()
    memory = model.encode(src)
    B = src.size(0)
    ys = torch.full((B, 1), SOS_IDX, dtype=torch.long, device=src.device)
    finished = torch.zeros(B, dtype=torch.bool, device=src.device)
    for _ in range(max_len):
        logits = model.decode(ys, memory, src)
        next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)
        next_tok = torch.where(finished.unsqueeze(1), torch.tensor(PAD_IDX, device=src.device), next_tok)
        ys = torch.cat([ys, next_tok], dim=1)
        finished = finished | (next_tok.squeeze(1) == EOS_IDX)
        if finished.all():
            break
    return ys[:, 1:]  # bỏ <sos>


def trim(seq: list[int]) -> list[int]:
    out = []
    for t in seq:
        if t == EOS_IDX:
            break
        if t not in (PAD_IDX, SOS_IDX):
            out.append(t)
    return out


def evaluate(model: Transformer, n: int = 5) -> float:
    """Trả về accuracy theo chuỗi (exact match) trên `n` ví dụ ngẫu nhiên."""
    src, _, tgt_out = make_batch(n, DEVICE)
    pred = greedy_decode(model, src, max_len=MAX_LEN + 2)
    correct = 0
    for i in range(n):
        gold = trim(tgt_out[i].tolist())
        got = trim(pred[i].tolist())
        if gold == got:
            correct += 1
        if i < 3:
            src_clean = [t for t in src[i].tolist() if t != PAD_IDX]
            print(f"  src={src_clean}  gold={gold}  pred={got}  {'OK' if gold == got else 'X'}")
    return correct / n


def predict_one(model: Transformer, seq: list[int]) -> list[int]:
    """Sinh kết quả cho một chuỗi đơn cụ thể (để demo những test case cố định)."""
    src = torch.tensor([seq], dtype=torch.long, device=DEVICE)
    pred = greedy_decode(model, src, max_len=len(seq) + 2)
    return trim(pred[0].tolist())


def final_demo(model: Transformer) -> None:
    """Chạy thử vài test case cố định + đo accuracy trên batch lớn ngẫu nhiên."""
    print("\n" + "=" * 60)
    print("DEMO sau khi train")
    print("=" * 60)

    # 1) Test case cố định — giúp user thấy hành vi ổn định, không phụ thuộc seed batch
    fixed_cases: list[list[int]] = [
        [3, 4, 5, 6, 7],
        [10, 9, 8, 7, 6, 5],
        [14, 3, 14, 3, 14],
        [5, 5, 5, 5],
        [3, 14, 7, 11, 4, 9, 8],
    ]
    print("\n[Test case cố định]")
    for s in fixed_cases:
        gold = list(reversed(s))
        got = predict_one(model, s)
        ok = "OK" if got == gold else "X "
        print(f"  {ok}  src={s}  →  pred={got}  (gold={gold})")

    # 2) Accuracy trên batch ngẫu nhiên lớn để có con số tin cậy
    print("\n[Accuracy trên 200 ví dụ ngẫu nhiên]")
    src, _, tgt_out = make_batch(200, DEVICE)
    pred = greedy_decode(model, src, max_len=MAX_LEN + 2)
    correct = sum(
        trim(pred[i].tolist()) == trim(tgt_out[i].tolist()) for i in range(src.size(0))
    )
    print(f"  exact-match acc = {correct}/{src.size(0)} = {correct / src.size(0):.3f}")


def main() -> None:
    set_seed(SEED)
    print(f"Reverse task | vocab={VOCAB_SIZE} | seq_len {MIN_LEN}..{MAX_LEN} | device={DEVICE}")

    model = Transformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_encoder_layers=ENC_LAYERS,
        num_decoder_layers=DEC_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_LEN + 4,
        dropout=0.1,
        pad_idx=PAD_IDX,
    ).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    for step in range(1, STEPS + 1):
        model.train()
        src, tgt_in, tgt_out = make_batch(BATCH, DEVICE)
        logits = model(src, tgt_in)
        loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), tgt_out.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % EVAL_EVERY == 0 or step == 1:
            print(f"\nstep {step:5d}  loss {loss.item():.4f}")
            acc = evaluate(model, n=10)
            print(f"  exact-match acc = {acc:.2f}")

    final_demo(model)


if __name__ == "__main__":
    main()
