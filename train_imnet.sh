RUN_NAME=imnet_10k_centroids

TOTAL_UPDATES=50000    # Total number of training steps
WARMUP_UPDATES=20000    # Warmup the learning rate over this many updates
PEAK_LR=0.0003         # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=32       # Number of sequences per batch (batch size)
UPDATE_FREQ=8          # Increase the batch size x

EMBEDDING_DIM=512 # RoBerta parameters
FFN_EMB_DIM=2048
NUM_ATT_HEADS=8
ENCODER_LAYERS=16
DROPOUT=0.2
SAVE_INTERVAL=3
EMB_WEIGHTS=~/imnet_10k/centroids.pkl


python train.py  ~/imnet_10k_bin \
--run-name $RUN_NAME \
--task masked_frame_lm --criterion masked_frame_lm \
--arch roberta_base \
--sample-break-mode eos \
--tokens-per-sample $TOKENS_PER_SAMPLE \
--optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
--lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
--dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
--max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
--max-update $TOTAL_UPDATES --log-format simple --log-interval 1 \
--encoder-embed-dim $EMBEDDING_DIM --encoder-ffn-embed-dim $FFN_EMB_DIM --encoder-attention-heads $NUM_ATT_HEADS --encoder-layers $ENCODER_LAYERS \
--no-epoch-checkpoints \
--dropout $DROPOUT --attention-dropout $DROPOUT --activation-dropout $DROPOUT \
--save-interval $SAVE_INTERVAL \
--num-workers 0 \
--emb-weights $EMB_WEIGHTS --freeze-emb
