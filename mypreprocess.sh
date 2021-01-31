PROCESSED_DIR=mydata
VOCAB_DIR=PhoBERT_base_fairseq
fairseq-preprocess --source-lang src --target-lang trg --trainpref $PROCESSED_DIR/train --validpref $PROCESSED_DIR/valid --testpref $PROCESSED_DIR/test --destdir $PROCESSED_DIR/bin --srcdict $VOCAB_DIR/dict.txt --tgtdict $VOCAB_DIR/dict.txt 