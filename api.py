import re
from collections import namedtuple
import torch
from transformers import AutoTokenizer
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import encoders
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq_cli.generate import get_symbols_to_strip_from_output
Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
import math

def buffered_read(lines_of_text, buffer_size):
  for line in lines_of_text:
    yield line

def make_batches(lines, cfg, task, max_positions, encode_fn):
    def encode_fn_target(x):
        return encode_fn(x)

    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]

    lengths = [t.numel() for t in tokens]
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(
            tokens, lengths),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=max_positions,
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        ids = batch["id"]
        src_tokens = batch["net_input"]["src_tokens"]
        src_lengths = batch["net_input"]["src_lengths"]
        

        yield Batch(
            ids=ids,
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            
        )



class FairseqRunner:
  def __init__(self):
    parser = options.get_interactive_generation_parser()
    args = options.parse_args_and_arch(parser)
    cfg = convert_namespace_to_omegaconf(args)
    utils.import_user_module(cfg.common)

    if cfg.interactive.buffer_size < 1:
        cfg.interactive.buffer_size = 1
    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.batch_size = 1

    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        not cfg.dataset.batch_size
        or cfg.dataset.batch_size <= cfg.interactive.buffer_size
    ), "--batch-size cannot be larger than --buffer-size"

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(cfg.task)

    # Load ensemble
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Initialize generator
    generator = task.build_generator(models, cfg.generation)

    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(cfg.tokenizer)
    bpe = encoders.build_bpe(cfg.bpe)

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(cfg.generation.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(), *[model.max_positions() for model in models]
    )
    if cfg.interactive.buffer_size > 1:
        logger.info("Sentence buffer size: %s", cfg.interactive.buffer_size)
    
    self.context = {
      'bpe': bpe,
      'tokenizer': tokenizer,
      'cfg': cfg,
      'task': task,
      'max_positions': max_positions,
      'use_cuda': use_cuda,
      'generator': generator,
      'models': models,
      'src_dict': src_dict,
      'tgt_dict': tgt_dict,
      'align_dict': align_dict,
    }

  def infer(self, lines_of_text):
    context = self.context

    bpe = context['bpe']
    tokenizer = context['tokenizer']
    cfg = context['cfg']
    task = context['task']
    max_positions = context['max_positions']
    use_cuda = context['use_cuda']
    generator = context['generator']
    models = context['models']
    src_dict = context['src_dict']
    tgt_dict = context['tgt_dict']
    align_dict = context['align_dict']

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    start_id = 0
    for inputs in [lines_of_text]:
        results = []
        for batch in make_batches(inputs, cfg, task, max_positions, encode_fn):
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            if use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()
            sample = {
                "net_input": {
                    "src_tokens": src_tokens,
                    "src_lengths": src_lengths,
                },
            }
            translations = task.inference_step(generator, models, sample)
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                results.append(
                    (
                        start_id + id,
                        src_tokens_i,
                        hypos,
                    )
                )

        # sort output to match input order
        for id_, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
            if src_dict is not None:
                src_str = src_dict.string(src_tokens, cfg.common_eval.post_process)

            # Process top predictions
            for hypo in hypos[: min(len(hypos), cfg.generation.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypo["alignment"],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=cfg.common_eval.post_process,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                )
                detok_hypo_str = decode_fn(hypo_str)
                score = hypo["score"] / math.log(2)  # convert to base 2
                #print(detok_hypo_str, hypo_str, hypo_tokens)
                yield (detok_hypo_str, hypo_str, hypo_tokens)
                
        # update running id_ counter
        start_id += len(inputs)

if __name__ == '__main__':
  from flask import Flask, escape, request

  app = Flask(__name__)

  runner = FairseqRunner()

  @app.route('/', methods=['POST'])
  def hello():
    #if request.json is None or 'text' not in request.json:
    #  return { 'error': '"text" field in JSON payload is required'}, 400
    text = request.json.get('text')
    if not isinstance(text, list):
      return { 'error': '"text" is expected to be a list of texts pieces'}, 400

    summary = [s for s, hypo, tokens in runner.infer(text)]
    return { 'ok': True, 'text': text, 'summary': summary }

  app.run('0.0.0.0', 3000)