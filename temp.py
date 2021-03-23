from fairseq import checkpoint_utils
state = checkpoint_utils.load_checkpoint_to_cpu("PhoBERT_base_fairseq/model.pt")

from fairseq.models.roberta import RobertaModel
phobert = RobertaModel.from_pretrained('/home/hieu/PycharmProjects/fscustomize/PhoBERT_base_fairseq', checkpoint_file='model.pt')

print('a')