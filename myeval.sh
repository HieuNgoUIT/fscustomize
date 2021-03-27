MODEL_DIR=500k_roberta/bin
MYDIRECTORY=`pwd`
fairseq-generate \
    --path $MODEL_DIR/checkpoint_best.pt $MODEL_DIR \
    --beam 5 --source-lang src --target-lang trg --skip-invalid-size-inputs-valid-test
    #--pretrained-roberta-checkpoint-folder $MYDIRECTORY/PhoBERT_base_fairseq