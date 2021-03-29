MODEL_DIR=500k_roberta
MYDIRECTORY=`pwd`
CUDA_VISIBLE_DEVICES=1 fairseq-interactive \
    --path $MODEL_DIR/model/checkpoint_best.pt $MODEL_DIR/bin \
    --beam 5 --source-lang src --target-lang trg --skip-invalid-size-inputs-valid-test
    #--pretrained-roberta-checkpoint-folder $MYDIRECTORY/PhoBERT_base_fairseq