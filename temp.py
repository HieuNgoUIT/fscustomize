from fairseq import checkpoint_utils
# state = checkpoint_utils.load_checkpoint_to_cpu("PhoBERT_base_fairseq/model.pt")
# with open("state_dict_cp.txt", "w") as f:
#     for i in state["model"].keys():
#         f.write(i)
# #print(state["model"].keys()) #154
import json

from fairseq.models.roberta import RobertaModel
phobert = RobertaModel.from_pretrained('/mnt/D/fscustomize/PhoBERT_base_fairseq', checkpoint_file='model.pt')
md = phobert.model.encoder.dictionary.indices
with open("641dict.txt", "w") as f:
    for i in md:
        f.write(f"{i} {md[i]}\n")
print(md)

# with open("state_dict_model.txt", "w") as f:
#     for i in phobert.model.encoder.state_dict().keys():
#         f.write(i)


#print(phobert.model.encoder.state_dict().keys()) #202
# with open('phobert.txt','w') as f:
#     f.write(str(phobert.state_dict))
# with open('phobert_encoder.txt','w') as f:
#     f.write(str(phobert.model.encoder.state_dict))
print('a')