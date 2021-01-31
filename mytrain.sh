PROCESSED_DIR=mydata
MODEL_DIR=mydata/model

fairseq-train $PROCESSED_DIR/bin \
    --save-dir $MODEL_DIR \
    --arch transformer_from_pretrained_roberta \
    --pretrained-roberta-checkpoint PhoBERT_base_fairseq/model.pt \
    --max-tokens 4096 \
    --optimizer adam \
    -s src \
    -t trg \
    --criterion label_smoothed_cross_entropy \
    --max-epoch 30 \
    --task translation_from_pretrained_roberta