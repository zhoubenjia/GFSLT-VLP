from huggingface_hub import snapshot_download

snapshot_download(repo_id="facebook/mbart-large-cc25", allow_patterns=["*.json", "pytorch_model.bin", "vocab.txt", '*.model'], cache_dir="./pretrain_models/MBart_proun/")