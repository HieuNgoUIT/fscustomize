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
        return TransformerEncoderFromPretrainedRoberta(args, src_dict, embed_tokens)

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


class TransformerEncoderFromPretrainedRoberta(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        assert hasattr(args, "pretrained_roberta_checkpoint"), (
            "--pretrained-roberta-checkpoint must be specified to load Transformer "
            "encoder from pretrained roberta"
        )
        roberta_loaded_state_dict = upgrade_state_dict_with_roberta_weights(
            state_dict=self.state_dict(),
            pretrained_roberta_checkpoint=args.pretrained_roberta_checkpoint,
        )
        self.load_state_dict(roberta_loaded_state_dict, strict=True)



@register_model_architecture(
    "transformer_from_pretrained_roberta", "transformer_from_pretrained_roberta"
)
def base_architecture(args):
    transformer_base_architecture(args)
