"""linuxembedmodel — a code embedding model trained from scratch on the Linux kernel.

Pipeline stages, in order:
    corpus          extract clean C/H text from the kernel tree
    tokenizer_train train a byte-level BPE vocabulary on that corpus
    pack            tokenize the corpus into a flat uint16 stream
    pretrain        MLM-pretrain a BERT encoder from random init
    mine_pairs      mine NL->C pairs from kernel-doc comments
    train_embed     contrastive training (InfoNCE, then GIST + InfoNCE)
    evaluate        retrieval evaluation on held-out kernel pairs
"""

__version__ = "0.1.0"
