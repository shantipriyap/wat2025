language = "hindi"  # or odia, malayalam, bengali

model_id = f"OdiaGenAI/facebook-nllb-200-3.3B-finetuned-{language}"

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_id)

LANG_CODES = {
    "hindi": "hin_Deva",
    "bengali": "ben_Beng",
    "odia": "ory_Orya",
    "malayalam": "mal_Mlym",
    "english": "eng_Latn"
}

tokenizer.src_lang = LANG_CODES.get("english")
tokenizer.tgt_lang = LANG_CODES.get(language.lower())

source_text = "This is a simple sentence to translate."

model_inputs = tokenizer(source_text, return_tensors="pt", padding=True, truncation=True).to(device)
generated_tokens = model.generate(
      **model_inputs,
      forced_bos_token_id=tokenizer.convert_tokens_to_ids("hin_Deva")
  )
target_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
print(f"Source Text: {source_text}\nTranslated Text ({language}): {target_text}")

# Result for Hindi below
# Source Text: This is a simple sentence to translate.
# Translated Text (hindi): यह अनुवाद करने के लिए एक सरल वाक्य है।