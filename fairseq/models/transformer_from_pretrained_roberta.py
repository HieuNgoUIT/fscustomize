# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict

from fairseq import checkpoint_utils
from fairseq.data.legacy.masked_lm_dictionary import MaskedLMDictionary
from fairseq.models import register_model, register_model_architecture
from fairseq.models.roberta.model import RobertaEncoder
from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerEncoder,
    TransformerModel,
    base_architecture as transformer_base_architecture,
)

from fairseq.models.roberta import RobertaModel

@register_model("transformer_from_pretrained_roberta")
class TransformerFromPretrainedRobertaModel(TransformerModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        parser.add_argument(
            "--pretrained-roberta-checkpoint",
            type=str,
            metavar="STR",
            help="roberta model to use for initializing transformer encoder",
        )

    @classmethod
    def build_model(self, args, task):
        assert hasattr(args, "pretrained_roberta_checkpoint"), (
            "You must specify a path for --pretrained-roberta-checkpoint to use "
            "--arch transformer_from_pretrained_roberta"
        )

        return super().build_model(args, task)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        args.max_positions = 256
        base_architecture(args)
        #return TransformerEncoderFromPretrainedRoberta(args, src_dict, embed_tokens)
        return WrapperEncoderOfRobertaModel(args, src_dict)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(args, tgt_dict, embed_tokens)


def upgrade_state_dict_with_roberta_weights(
    state_dict: Dict[str, Any], pretrained_roberta_checkpoint: str
) -> Dict[str, Any]:
    """
    Load roberta weights into a Transformer encoder.

    Args:
        state_dict: state dict for either TransformerEncoder 
        pretrained_roberta_checkpoint: checkpoint to load XLM weights from

    Raises:
        AssertionError: If architecture (num layers, attention heads, etc.)
            does not match between the current Transformer encoder or
            decoder and the pretrained_xlm_checkpoint
    """
    if not os.path.exists(pretrained_roberta_checkpoint):
        raise IOError("Model file not found: {}".format(pretrained_roberta_checkpoint))

    state = checkpoint_utils.load_checkpoint_to_cpu(pretrained_roberta_checkpoint)
    roberta_state_dict = state["model"]
    for key in roberta_state_dict.keys():
        for search_key in ["embed_tokens", "embed_positions", "layers"]:
            if search_key in key:
                subkey = key[key.find(search_key) :]
                assert subkey in state_dict, (
                    "{} Transformer encoder / decoder "
                    "state_dict does not contain {}. Cannot "
                    "load {} from pretrained Roberta checkpoint "
                    "{} into Transformer.".format(
                        str(state_dict.keys()), subkey, key, pretrained_roberta_checkpoint
                    )
                )

                state_dict[subkey] = roberta_state_dict[key]
    return state_dict


def copy_state_dict_with_pretrained_roberta(state_dict: Dict[str, Any], pretrained_roberta_checkpoint: str
) -> Dict[str, Any]:
    state = checkpoint_utils.load_checkpoint_to_cpu(pretrained_roberta_checkpoint)
    roberta_state_dict = state["model"]
    state_dict = roberta_state_dict
    return state_dict

class WrapperEncoderOfRobertaModel(RobertaEncoder):
    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)
        self.padding_idx = dictionary.pad()
        phobert = RobertaModel.from_pretrained('/mnt/D/fscustomize/PhoBERT_base_fairseq', checkpoint_file='model.pt')
        #self.load_state_dict(phobert.model.encoder.state_dict(), strict=False)
    def forward(
            self,
            src_tokens,
            features_only=False,
            return_all_hiddens=False,
            masked_tokens=None,
            **unused
    ):
        x, extra = super().forward(src_tokens=src_tokens, return_all_hiddens=return_all_hiddens, features_only=True)
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],
            "encoder_embedding": None,
            "encoder_states": None,
            "src_tokens": [],
            "src_lengths": [],
        }


class TransformerEncoderFromPretrainedRoberta(RobertaEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary)
        assert hasattr(args, "pretrained_roberta_checkpoint"), (
            "--pretrained-roberta-checkpoint must be specified to load Transformer "
            "encoder from pretrained roberta"
        )
        self.padding_idx = dictionary.pad()
        roberta_loaded_state_dict = upgrade_state_dict_with_roberta_weights(
           state_dict=self.state_dict(),
           pretrained_roberta_checkpoint=args.pretrained_roberta_checkpoint,
        )
        self.load_state_dict(roberta_loaded_state_dict, strict=True)

    def forward(
            self,
            src_tokens,
            features_only=False,
            return_all_hiddens=False,
            masked_tokens=None,
            **unused
    ):
        x, extra = super().forward(src_tokens=src_tokens, return_all_hiddens=return_all_hiddens, features_only=True)
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],
            "encoder_embedding": None,
            "encoder_states": None,
            "src_tokens": [],
            "src_lengths": [],
        }

@register_model_architecture(
    "transformer_from_pretrained_roberta", "transformer_from_pretrained_roberta"
)
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
    args.untie_weights_roberta = getattr(args, "untie_weights_roberta", False)
    args.spectral_norm_classification_head = getattr(
        args, "spectral_norm_classification_head", False
    )
    #transformer_base_architecture(args)
