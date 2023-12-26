
from transformers import MBartForConditionalGeneration, MBartTokenizer, MBartConfig

from hftrim.ModelTrimmers import MBartTrimmer

import utils
from hftrim.TokenizerTrimmer import TokenizerTrimmer


raw_data = utils.load_dataset_file('data/Phonexi-2014T/labels.train')

data = []

for key,value in raw_data.items():
    sentence = value['text']
    # gloss = value['gloss']
    data.append(sentence)
    # data.append(gloss.lower())

tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="de_DE", tgt_lang="de_DE")

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
configuration = model.config

# trim tokenizer
tt = TokenizerTrimmer(tokenizer)
tt.make_vocab(data)
tt.make_tokenizer()

# trim model
mt = MBartTrimmer(model, configuration, tt.trimmed_tokenizer)
mt.make_weights(tt.trimmed_vocab_ids)
mt.make_model()

new_tokenizer = tt.trimmed_tokenizer
new_model = mt.trimmed_model

new_tokenizer.save_pretrained('pretrain_models/MBart_trimmed')
new_model.save_pretrained('pretrain_models/MBart_trimmed')

## mytran_model
configuration = MBartConfig.from_pretrained('pretrain_models/mytran/config.json')
configuration.vocab_size = new_model.config.vocab_size
mytran_model = MBartForConditionalGeneration._from_config(config=configuration)
mytran_model.model.shared = new_model.model.shared

mytran_model.save_pretrained('pretrain_models/mytran/')











