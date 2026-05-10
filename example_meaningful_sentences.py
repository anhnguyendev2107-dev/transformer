"""
Ví dụ copy task với câu tiếng Việt có nghĩa (từ vựng nhỏ, tách từ bằng khoảng trắng).

Chạy: python example_meaningful_sentences.py
"""

from __future__ import annotations

import random
from typing import Sequence

import torch
import torch.nn as nn
import torch.optim as optim

from transformer import Transformer

PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2

# Câu mẫu (thêm/bớt câu trong list này để thử)
CORPUS: list[str] = [
    "xin chào thế giới",
    "học máy là thú vị",
    "transformer dùng cơ chế attention",
    "mô hình ngôn ngữ dự đoán từ tiếp theo",
    "python là ngôn ngữ lập trình phổ biến",
    "tensor là mảng đa chiều trong torch",
    "encoder đọc đầu vào thành biểu diễn ẩn",
    "decoder sinh từng từ một cách tự hồi quy",
    "mask che các vị trí padding khi tính attention",
    "học sâu cần gpu để huấn luyện nhanh",
    "dữ liệu tốt giúp mô hình khái quát hơn",
    "gradient descent cập nhật trọng số mạng",
    "batch norm ổn định huấn luyện mạng sâu",
    "overfitting xảy ra khi mô hình nhớ bài",
    "token là đơn vị nhỏ nhất khi xử lý câu",
]

D_MODEL = 128
NUM_HEADS = 8
ENC_LAYERS = DEC_LAYERS = 2
D_FF = 512
MAX_LEN = 32
BATCH = 8
STEPS = 4000
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_vocab(sentences: Sequence[str]) -> tuple[dict[str, int], dict[int, str], int]:
    words: set[str] = set()
    for s in sentences:
        for w in s.lower().split():
            words.add(w)
    sorted_words = sorted(words)
    stoi: dict[str, int] = {"<pad>": PAD_IDX, "<sos>": SOS_IDX, "<eos>": EOS_IDX}
    for w in sorted_words:
        stoi[w] = len(stoi)
    itos = {i: t for t, i in stoi.items()}
    return stoi, itos, len(stoi)


def encode(sentence: str, stoi: dict[str, int]) -> list[int]:
    return [stoi[w] for w in sentence.lower().split()]


def decode(ids: list[int], itos: dict[int, str]) -> str:
    return " ".join(itos.get(i, "?") for i in ids if i not in (PAD_IDX, SOS_IDX, EOS_IDX))


def pad_sequences(seqs: list[list[int]], pad: int) -> torch.Tensor:
    L = max(len(s) for s in seqs)
    out = torch.full((len(seqs), L), pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def batch_from_corpus(
    corpus_ids: list[list[int]],
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Một batch: chọn ngẫu nhiên các câu, pad; tgt_in / tgt_out giống train_copy_task."""
    picked = random.choices(corpus_ids, k=batch_size)
    lengths = [len(s) for s in picked]
    L = max(lengths)
    src = torch.full((batch_size, L), PAD_IDX, dtype=torch.long, device=device)
    for i, s in enumerate(picked):
        src[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)

    tgt_in = torch.full((batch_size, L + 1), PAD_IDX, dtype=torch.long, device=device)
    tgt_out = torch.full((batch_size, L + 1), PAD_IDX, dtype=torch.long, device=device)
    for i, s in enumerate(picked):
        ln = len(s)
        tgt_in[i, 0] = SOS_IDX
        tgt_in[i, 1 : ln + 1] = torch.tensor(s, dtype=torch.long, device=device)
        tgt_out[i, :ln] = torch.tensor(s, dtype=torch.long, device=device)
        tgt_out[i, ln] = EOS_IDX
    return src, tgt_in, tgt_out


def main() -> None:
    set_seed(SEED)
    stoi, itos, vocab_size = build_vocab(CORPUS)
    corpus_ids = [encode(s, stoi) for s in CORPUS]

    print("Một vài câu trong corpus:")
    for s in CORPUS[:3]:
        print(" ", s)
    print(f"Vocab size: {vocab_size} (gồm <pad>, <sos>, <eos> và từng từ)\n")

    model = Transformer(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_encoder_layers=ENC_LAYERS,
        num_decoder_layers=DEC_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_LEN + 4,
        dropout=0.1,
        pad_idx=PAD_IDX,
    ).to(DEVICE)
    params = model.parameters()
    opt = optim.Adam(params, lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    model.train()
    for step in range(1, STEPS + 1):
        src, tgt_in, tgt_out = batch_from_corpus(corpus_ids, BATCH, DEVICE)
        logits = model(src, tgt_in)
        loss = loss_fn(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 500 == 0 or step == 1:
            print(f"step {step:5d}  loss {loss.item():.4f}")

    # Thử một câu đã có trong corpus (để xem mô hình “nhớ” copy)
    model.eval()
    test_sentence = CORPUS[1]
    ids = encode(test_sentence, stoi)
    L = len(ids)
    src = torch.full((1, L), PAD_IDX, dtype=torch.long, device=DEVICE)
    src[0, :L] = torch.tensor(ids, device=DEVICE)
    tgt_in = torch.full((1, L + 1), PAD_IDX, dtype=torch.long, device=DEVICE)
    tgt_in[0, 0] = SOS_IDX
    tgt_in[0, 1 : L + 1] = torch.tensor(ids, device=DEVICE)

    with torch.no_grad():
        logits = model(src, tgt_in)
        pred = logits.argmax(dim=-1)[0].tolist()

    # Bỏ pad và lấy tới EOS đầu tiên nếu có
    out_ids: list[int] = []
    for t in pred:
        if t == EOS_IDX:
            break
        if t not in (PAD_IDX, SOS_IDX):
            out_ids.append(t)
        if len(out_ids) >= L:
            break

    print("\n--- Kiểm tra copy (một câu trong corpus) ---")
    print("Gốc:   ", test_sentence)
    print("Dự đoán:", decode(out_ids, itos))


if __name__ == "__main__":
    main()
