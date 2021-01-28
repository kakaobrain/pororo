# Copyright (c) Facebook, Inc., its affiliates and Kakao Brain. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqDecoder,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.roberta import RobertaClassificationHead, RobertaLMHead
from fairseq.models.roberta.hub_interface import RobertaHubInterface
from fairseq.modules import TransformerSentenceEncoder
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from pororo.models.brainbert.utils import CustomChar
from pororo.tasks.utils.download_utils import download_or_load


class CharBrainLabertaHubInterface(RobertaHubInterface):

    def __init__(self, args, task, model):
        super().__init__(args, task, model)
        self.bpe = CustomChar()

    def tokenize(self, sentence: str, add_special_tokens: bool = False):
        result = self.bpe.encode(sentence)
        return f"<s> {result} </s>" if add_special_tokens else result

    def encode(
        self,
        sentence: str,
        *addl_sentences,
        add_special_tokens: bool = True,
        no_separator: bool = False,
    ) -> torch.LongTensor:
        bpe_sentence = self.tokenize(
            sentence,
            add_special_tokens=add_special_tokens,
        )

        for s in addl_sentences:
            bpe_sentence += " </s>" if not no_separator and add_special_tokens else ""
            bpe_sentence += (" " + self.tokenize(s, add_special_tokens=False) +
                             " </s>" if add_special_tokens else "")

        tokens = self.task.source_dictionary.encode_line(
            bpe_sentence,
            append_eos=False,
            add_if_not_exist=False,
        )
        return tokens.long()

    def extract_features(
        self,
        tokens: torch.LongTensor,
        segments: torch.LongTensor,
        return_all_hiddens: bool = False,
    ) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.size(-1) > self.model.max_positions():
            raise ValueError("tokens exceeds maximum length: {} > {}".format(
                tokens.size(-1), self.model.max_positions()))
        features, extra = self.model(
            tokens.to(device=self.device),
            segments.to(device=self.device),
            features_only=True,
            return_all_hiddens=return_all_hiddens,
        )
        if return_all_hiddens:
            # convert from T x B x C -> B x T x C
            inner_states = extra["inner_states"]
            return [inner_state.transpose(0, 1) for inner_state in inner_states]
        return features  # just the last layer's features

    def predict(
        self,
        head: str,
        tokens: torch.LongTensor,
        segments: torch.LongTensor,
        return_logits: bool = False,
    ):
        features = self.extract_features(
            tokens.to(device=self.device),
            segments.to(device=self.device),
        )
        logits = self.model.classification_heads[head](features)

        if return_logits:
            return logits
        return F.log_softmax(logits, dim=-1)

    @torch.no_grad()
    def predict_srl(
        self,
        sentence: str,
        segment: str,
    ):
        label_fn = lambda label: self.task.label_dictionary.string([label])

        bpe_sentence = f"<s> {sentence} </s>"
        tokens = self.task.source_dictionary.encode_line(
            bpe_sentence,
            append_eos=False,
            add_if_not_exist=False,
        ).long()

        bpe_segment = f"<s> {segment} </s>"
        segments = self.task.segment_dictionary.encode_line(
            bpe_segment,
            append_eos=False,
            add_if_not_exist=False,
        ).long()

        # Get first batch and ignore <s> & </s> tokens
        preds = (self.predict(
            "sequence_tagging_label_head",
            tokens,
            segments,
        )[0, 1:-1, :].argmax(dim=1).cpu().numpy())
        labels = [
            label_fn(int(pred) + self.task.label_dictionary.nspecial)
            for pred in preds
        ]
        return [(token, label) for token, label in zip(sentence.split(), labels)
               ]


@register_model("roberta_label")
class RobertaLabelModel(FairseqLanguageModel):

    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args
        # We follow BERT's random weight initialization
        self.apply(init_bert_params)
        self.classification_heads = nn.ModuleDict()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--num-segments",
            type=int,
            metavar="N",
            help="num segments",
        )
        parser.add_argument(
            "--encoder-layers",
            type=int,
            metavar="L",
            help="num encoder layers",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            metavar="D",
            help="dropout probability",
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-positions",
            type=int,
            help="number of positional embeddings to learn",
        )
        parser.add_argument(
            "--load-checkpoint-heads",
            action="store_true",
            help="(re-)register and load heads when loading checkpoints",
        )
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument(
            "--encoder-layerdrop",
            type=float,
            metavar="D",
            default=0,
            help="LayerDrop probability for encoder",
        )
        parser.add_argument(
            "--encoder-layers-to-keep",
            default=None,
            help="which layers to *keep* when pruning as a comma-separated list",
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaLabelEncoder(args, task.source_dictionary)
        return cls(args, encoder)

    def forward(
        self,
        src_tokens,
        seg_tokens,
        features_only=False,
        return_all_hiddens=False,
        classification_head_name=None,
        **kwargs,
    ):
        if classification_head_name is not None:
            features_only = True

        x, extra = self.decoder(
            src_tokens,
            seg_tokens,
            features_only,
            return_all_hiddens,
            **kwargs,
        )

        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)
        return x, extra

    def register_classification_head(
        self,
        name,
        num_classes=None,
        inner_dim=None,
        **kwargs,
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[
                name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                print(
                    'WARNING: re-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim,
                        prev_inner_dim))
        self.classification_heads[name] = RobertaClassificationHead(
            self.args.encoder_embed_dim,
            inner_dim or self.args.encoder_embed_dim,
            num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout,
        )

    @property
    def supported_targets(self):
        return {"self"}

    @classmethod
    def load_model(cls, model_name: str, lang: str, **kwargs):
        """
        Load pre-trained model as RobertaHubInterface.
        :param model_name: model name from available_models
        :return: pre-trained model
        """
        from fairseq import hub_utils

        # cache directory is treated as the home directory
        # for both model and data files
        ckpt_dir = download_or_load(model_name, lang)
        x = hub_utils.from_pretrained(
            ckpt_dir,
            "model.pt",
            ckpt_dir,
            **kwargs,
        )

        return CharBrainLabertaHubInterface(
            x["args"],
            x["task"],
            x["models"][0],
        )

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        prefix = name + "." if name != "" else ""
        current_head_names = ([] if not hasattr(self, "classification_heads")
                              else self.classification_heads.keys())

        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "classification_heads."):
                continue

            head_name = k[len(prefix + "classification_heads."):].split(".")[0]
            num_classes = state_dict[prefix + "classification_heads." +
                                     head_name + ".out_proj.weight"].size(0)
            inner_dim = state_dict[prefix + "classification_heads." +
                                   head_name + ".dense.weight"].size(0)

            if getattr(self.args, "load_checkpoint_heads", False):
                if head_name not in current_head_names:
                    self.register_classification_head(
                        head_name,
                        num_classes,
                        inner_dim,
                    )
            else:
                if head_name not in current_head_names:
                    print(
                        "WARNING: deleting classification head ({}) from checkpoint "
                        "not present in current model: {}".format(head_name, k))
                    keys_to_delete.append(k)
                elif (num_classes !=
                      self.classification_heads[head_name].out_proj.out_features
                      or inner_dim !=
                      self.classification_heads[head_name].dense.out_features):
                    print(
                        "WARNING: deleting classification head ({}) from checkpoint "
                        "with different dimensions than current model: {}".
                        format(head_name, k))
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, "classification_heads"):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "classification_heads." + k not in state_dict:
                    print("Overwriting", prefix + "classification_heads." + k)
                    state_dict[prefix + "classification_heads." + k] = v


class RobertaLabelEncoder(FairseqDecoder):
    """RoBERTa encoder.
    Implements the :class:`~fairseq.models.FairseqDecoder` interface required
    by :class:`~fairseq.models.FairseqLanguageModel`.
    """

    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.args = args

        # RoBERTa is a sentence encoder model, so users will intuitively trim
        # encoder layers. However, the implementation uses the fairseq decoder,
        # so we fix here.
        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
            args.decoder_layers_to_keep = args.encoder_layers_to_keep
            args.encoder_layers_to_keep = None

        self.sentence_encoder = TransformerSentenceEncoder(
            padding_idx=dictionary.pad(),
            vocab_size=len(dictionary),
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            layerdrop=args.encoder_layerdrop,
            max_seq_len=args.max_positions,
            num_segments=48 + 4,
            encoder_normalize_before=True,
            apply_bert_init=True,
            activation_fn=args.activation_fn,
        )
        self.lm_head = RobertaLMHead(
            embed_dim=args.encoder_embed_dim,
            output_dim=len(dictionary),
            activation_fn=args.activation_fn,
            weight=self.sentence_encoder.embed_tokens.weight,
        )

    def forward(
        self,
        src_tokens,
        seg_tokens,
        features_only=False,
        return_all_hiddens=False,
        masked_tokens=None,
        **unused,
    ):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states. Note that the hidden
                  states have shape `(src_len, batch, vocab)`.
        """
        x, extra = self.extract_features(
            src_tokens,
            seg_tokens,
            return_all_hiddens=return_all_hiddens,
        )
        if not features_only:
            x = self.output_layer(x, masked_tokens=masked_tokens)
        return x, extra

    def extract_features(
        self,
        src_tokens,
        seg_tokens,
        return_all_hiddens=False,
        **kwargs,
    ):
        inner_states, _ = self.sentence_encoder(
            src_tokens,
            seg_tokens,
            last_state_only=not return_all_hiddens,
            token_embeddings=kwargs.get("token_embeddings", None),
        )
        features = inner_states[-1].transpose(0, 1)  # T x B x C -> B x T x C
        return features, {
            "inner_states": inner_states if return_all_hiddens else None
        }

    def output_layer(self, features, masked_tokens=None, **unused):
        return self.lm_head(features, masked_tokens)

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions


@register_model_architecture("roberta_label", "roberta_label")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.0)


@register_model_architecture("roberta_label", "roberta_label_base")
def roberta_label_base_architecture(args):
    base_architecture(args)
