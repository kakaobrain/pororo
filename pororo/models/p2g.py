# Copyright (c) Kakao Brain. All Rights Reserved

import pickle

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

UNK_TOKEN = "<UNK>"  # 0
PAD_TOKEN = "<PAD>"  # 1


class P2gM(object):

    def __init__(self, pinyin2idx: str, char2idx: str, ckpt: str, device: str):

        self.pinyin2idx = pickle.load(open(pinyin2idx, "rb"))
        self.character2idx = pickle.load(open(char2idx, "rb"))
        self.idx2character = {
            idx: character for character, idx in self.character2idx.items()
        }

        self.model = Net(self.pinyin2idx, self.character2idx, device)
        self.model.load_state_dict(
            torch.load(ckpt, map_location=device)["state_dict"])
        self.model.eval().to(device)

    def __call__(self, text, batch_size=32):
        """
        num_batches = len(self.train_loader)
        n_steps = 1
        for epoch in range(1, self.args.epochs + 1):
            print("epoch {}/{} :".format(epoch, self.args.epochs), end="\r")
            start = time.time()
            self.model.train()
            for idx, batch in enumerate(self.train_loader, start=1):
                n_steps += 1
            xs : [b, t]
            lengths : [b, ]
        Returns
            logits: [b*t, n_classes]

        """
        if isinstance(text, str):
            sents = text.splitlines()
        else:
            sents = text
        dataloader = get_dataloader(
            self.pinyin2idx,
            self.character2idx,
            sents,
            batch_size,
        )

        outputs = []
        for i, batch in enumerate(dataloader, start=1):
            with torch.no_grad():
                xs, lengths, inputs = batch
                logits = self.model(xs, lengths)
                preds = torch.argmax(logits, -1).detach().cpu().numpy()

                for inp, pred in zip(inputs, preds):
                    pinyins = inp.split()
                    pred = pred[:len(pinyins)]
                    characters = [self.idx2character[idx] for idx in pred]
                    characters = [
                        p if c == "<UNK>" else c
                        for c, p in zip(characters, pinyins)
                    ]
                    outputs.append("".join(characters))

        return outputs


class Net(nn.Module):

    def __init__(self, pinyin2idx, char2idx, device):
        super(Net, self).__init__()
        self.pinyin2idx = pinyin2idx
        self.character2idx = char2idx
        self.vocab_size = len(self.pinyin2idx)
        self.num_classes = len(self.character2idx)
        self.embedding_size = 512
        self.hidden_size = 512
        self.num_layers = 2

        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embedding_size,
            padding_idx=0,
        )  # Do NOT change!

        self.lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
            num_layers=self.num_layers,
        )

        self.logit_layer = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_classes),
        )
        self.device = device

    def forward(self, xs, lengths):
        """
            xs : [b, t]
            lengths : [b, ]
        Returns
            logits: [b*t, n_classes]
        """
        xs = xs.to(self.device)

        emb = self.embedding(xs)
        seqlen = xs.size(1)
        packed = pack_padded_sequence(
            emb,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(packed)

        hidden, _ = pad_packed_sequence(
            outputs,
            batch_first=True,
            total_length=seqlen,
        )
        logits = self.logit_layer(hidden)
        return logits


class P2gDataset(Dataset):

    def __init__(self, pinyin2idx, character2idx, sents):
        self.inputs = sents
        self.num_samples = len(self.inputs)
        self.pinyin2idx = pinyin2idx
        self.character2idx = character2idx

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        inp = self.inputs[idx].strip()
        pinyins = inp.split()
        if len(pinyins) == 0:
            return None
        unk_id = self.pinyin2idx[UNK_TOKEN]
        x = [self.pinyin2idx.get(pinyin, unk_id) for pinyin in pinyins]
        x = torch.tensor(x, dtype=torch.long)
        return x, inp


def collate_fn(data):
    """
    :param data:
    :return: padded inputs and outputs
    """
    xs, inputs = zip(*data)

    # pad sequences
    lengths = [len(x) for x in xs]  # (b,)
    xs_ = torch.zeros(len(lengths), max(lengths)).long()
    for i in range(len(lengths)):
        end = lengths[i]
        x = xs[i]
        xs_[i, :end] = x[:end]

    return xs_, lengths, inputs


def get_dataloader(pinyin2idx, character2idx, sents, batch_size):
    dataset = P2gDataset(pinyin2idx, character2idx, sents)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return dataloader
