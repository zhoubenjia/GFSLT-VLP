## How to Use MBart Model Weights？

The ./pretrain_models/ directory contains two subdirectories: MBart_proun and mytran. The former comprises the [facebook/mbart-large-cc25](https://huggingface.co/facebook/mbart-large-cc25) model that we downloaded from [HuggingFace](https://huggingface.co). Its encoder is utilized to extract sentence features. The latter contains the configuration files for our GFSLT model, also built upon the MBart architecture. Nevertheless, certain weight parameters in it, such as the Word Embedding layer of the Text Decoder, are also inherited from [facebook/mbart-large-cc25](https://huggingface.co/facebook/mbart-large-cc25).

You can obtain the weight parameters through the following two methods:

* Directly download the pre-processed model from [Baidu Netdisk](https://pan.baidu.com/s/15h9dsHMPH8dXH7glZvZnng?pwd=4s1p) using the extraction code: 4s1p.
* python trim_model.py

The file directory structure should be as follows:

```
pretrain_models/
├──MBart_trimmed/
│  |── config.json
│  |── pytorch_model.bin
│  |── sentencepiece.bpe.model
│  |── special_tokens_map.json
│  |── tokenizer_config.json
├──mytran/
│  |── config.json
│  |── pytorch_model.bin
```

After completing these steps, congratulations! You are ready to begin training your model.