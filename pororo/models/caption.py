# Copyright (c) Facebook, Inc. and its affiliates and Kakao Brain. All Rights Reserved

"""
DETR Transformer class.
Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import List, Optional, Tuple

import requests
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch import Tensor, nn


class Transformer(nn.Module):

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_position=128,
        vocab_size=30522,
        pad_token_id=0,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            encoder_norm,
        )

        self.embeddings = DecoderEmbeddings(
            d_model,
            max_position,
            dropout,
            vocab_size,
            pad_token_id,
        )

        decoder_layer = TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
        )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(
            mask == 1, float(0.0)))
        return mask

    def forward(self, feature, location, tgt, tgt_pad_mask):
        """
        The size of image feature is fixed by DeTR:
            feature:  [bsz, 100, 256]
            location: [bsz, 100, 4]

        Caption size is as below:
            tgt:          [bsz, max_positions]
            tgt_pad_mask: [bsz, max_positions]

        """
        feature = feature.permute(1, 0, 2)

        memory = self.encoder(feature, location)
        tgt = self.embeddings(tgt).permute(1, 0, 2)

        hs = self.decoder(
            tgt,
            memory,
            tgt_mask=self.generate_square_subsequent_mask(len(tgt)).to(
                tgt.device),
            tgt_key_padding_mask=tgt_pad_mask,
        )
        return hs


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = norm

    def forward(self, feature, location):
        output = feature

        for layer in self.layers:
            output = layer(output, location)

        return self.norm(output)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.location_embedding = nn.Linear(4, d_model)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def with_location_embed(self, tensor, location):
        location_embed = self.location_embedding(location).permute(1, 0, 2)
        return tensor + location_embed

    def forward(self, feature, location):
        src2 = self.norm1(feature)
        q = k = self.with_location_embed(src2, location)
        src2 = self.self_attn(q, k, value=src2)[0]
        feature = feature + self.dropout1(src2)

        src2 = self.norm2(feature)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        feature = feature + self.dropout2(src2)
        return feature


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.norm = norm

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
    ):
        output = tgt

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )

        return self.norm(output)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = tgt2
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=tgt2,
            key=memory,
            value=memory,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class DecoderEmbeddings(nn.Module):

    def __init__(
        self,
        d_model,
        max_position,
        dropout,
        vocab_size,
        pad_token_id,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_token_id,
        )
        self.position_embeddings = nn.Embedding(max_position, d_model)

        self.layernorm = torch.nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        input_shape = x.size()
        seq_length = input_shape[1]
        device = x.device

        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(
            input_shape)  # bsz x max_pos

        input_embeds = self.word_embeddings(x)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = input_embeds + position_embeds
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args["hidden_dim"],
        dropout=args["dropout"],
        nhead=args["nheads"],
        dim_feedforward=args["dim_feedforward"],
        num_encoder_layers=args["enc_layers"],
        num_decoder_layers=args["dec_layers"],
        max_position=args["max_position"],
        vocab_size=args["vocab_size"],
        pad_token_id=args["pad_token_id"],
    )


class Caption(nn.Module):

    def __init__(self, pad_token_id: int, vocab_size: int):
        super().__init__()
        config = {
            "hidden_dim": 256,
            "dropout": 0.1,
            "nheads": 8,
            "dim_feedforward": 1024,
            "enc_layers": 3,
            "dec_layers": 6,
            "max_position": 128,
            "pad_token_id": pad_token_id,
            "vocab_size": vocab_size,
        }

        self.transformer = build_transformer(config)
        self.mlp = nn.Linear(
            config["hidden_dim"],
            config["vocab_size"],
            bias=False,
        )

    def forward(self, feature, location, target, target_mask):
        hs = self.transformer(feature, location, target, target_mask)
        out = self.mlp(hs.permute(1, 0, 2))
        return out


class NestedTensor(object):

    def __init__(self, tensors, mask: Optional[torch.Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[torch.Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, _, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
            m[:img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class Detr(object):

    def __init__(self, device: str):
        self.device = device

        self.model = torch.hub.load(
            "facebookresearch/detr",
            "detr_resnet50",
            pretrained=True,
        )
        self.model.eval().to(device)

        # standard PyTorch mean-std input image normalization
        self.transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(1)
        b = [
            (x_c - 0.5 * w),
            (y_c - 0.5 * h),
            (x_c + 0.5 * w),
            (y_c + 0.5 * h),
        ]
        return torch.stack(b, dim=1).to(self.device)

    def rescale_bboxes(self, out_bbox: torch.Tensor, size: Tuple[int, int]):
        width, height = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = (b * torch.tensor(
            [
                width,
                height,
                width,
                height,
            ],
            dtype=torch.float32,
        ).to(self.device))
        return b.to(self.device)

    def extract_feature(self, img: str, threshold: float = 0.9):
        try:
            im = Image.open(img).convert("RGB")
        except:
            im = Image.open(requests.get(img, stream=True).raw).convert("RGB")

        # mean-std normalize the input image (batch-size: 1)
        img = self.transform(im).unsqueeze(0).to(self.device)

        if isinstance(img, (list, torch.Tensor)):
            img = nested_tensor_from_tensor_list(img)

        # propagate through the model
        features, pos = self.model.backbone(img)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.model.transformer(
            self.model.input_proj(src),
            mask,
            self.model.query_embed.weight,
            pos[-1],
        )[0]

        outputs_class = self.model.class_embed(hs)
        outputs_coord = self.model.bbox_embed(hs).sigmoid()

        probas = outputs_class[-1][0, :, :-1]
        # we can keep only predictions with defined confidence
        # keep = probas.max(-1).values > threshold

        # convert boxes from [0; 1] to image scales
        bboxes_scaled = self.rescale_bboxes(outputs_coord[-1][0], im.size)

        return {
            "features": hs[-1][0, :, :],
            "logits": probas,
            "boxes": bboxes_scaled,
        }
