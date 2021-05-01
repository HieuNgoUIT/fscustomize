PROCESSED_DIR=mydata
MYDIRECTORY=`pwd`

#CUDA_VISIBLE_DEVICES=0
fairseq-train $PROCESSED_DIR/bin \
    --save-dir $PROCESSED_DIR/model \
    --arch transformers2 \
    --bert-model-name bert-base-multilingual-cased \
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


#mydata/bin --save-dir mydata/model --arch transformers2 --bert-model-name bert-base-multilingual-cased --share-decoder-input-output-embed --optimizer adam --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096 --skip-invalid-size-inputs-valid-test