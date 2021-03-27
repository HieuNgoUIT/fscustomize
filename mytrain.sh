PROCESSED_DIR=500k_roberta
MODEL_DIR=500k_roberta/model
MYDIRECTORY=`pwd`
#fairseq-train $PROCESSED_DIR/bin \
#    --save-dir $MODEL_DIR \
#    --arch transformer_from_pretrained_roberta \
#    --pretrained-roberta-checkpoint PhoBERT_base_fairseq/model.pt \
#    --pretrained-roberta-checkpoint-folder $MYDIRECTORY/PhoBERT_base_fairseq \
#    --max-tokens 4096 \
#    --optimizer adam \
#    -s src \
#    -t trg \
#    --criterion label_smoothed_cross_entropy \
#    --max-epoch 30 \
#    --task translation_from_pretrained_roberta \
#    --share-decoder-input-output-embed --skip-invalid-size-inputs-valid-test
fairseq-train $PROCESSED_DIR/bin \
    --save-dir $MODEL_DIR \
    --arch transformer \
    --pretrained-roberta-checkpoint-folder $MYDIRECTORY/PhoBERT_base_fairseq \
    --share-decoder-input-output-embed \
    --optimizer adam \
    --clip-norm 0.0 \
    --lr 5e-4 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --dropout 0.3 \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 --max-tokens 4096 \




#mydata/bin --arch transformer --share-decoder-input-output-embed --optimizer adam --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096

#mydata/bin --save-dir mydata/model --arch transformer_from_pretrained_roberta --pretrained-roberta-checkpoint PhoBERT_base_fairseq/model.pt --max-tokens 4096 --optimizer adam -s src -t trg --criterion label_smoothed_cross_entropy --max-epoch 30 --task translation_from_pretrained_roberta --pretrained-roberta-checkpoint-folder /mnt/D/fscustomize/PhoBERT_base_fairseq
#mydata/bin --arch transformer --share-decoder-input-output-embed --optimizer adam --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096 --eval-bleu --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples --best-checkpoint-metric bleu --maximize-best-checkpoint-metric

#data-bin/iwslt14.tokenized.de-en --arch transformer_iwslt_de_en --share-decoder-input-output-embed --optimizer adam --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096 --eval-bleu --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples --best-checkpoint-metric bleu --maximize-best-checkpoint-metric