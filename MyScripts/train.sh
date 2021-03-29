PROCESSED_DIR=500k_roberta
MYDIRECTORY=`pwd`

#CUDA_VISIBLE_DEVICES=0
fairseq-train $PROCESSED_DIR/bin \
    --save-dir $PROCESSED_DIR/model \
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
    --skip-invalid-size-inputs-valid-test
