languages = ["bengali", "hindi", "malayalam", "odia"]
from datasets import load_dataset

# df1 = load_dataset(wat_dataset_id, split = "train").to_pandas()[["english", language]]
#Rename cols to "src" and "tgt"
# df1 = df1.rename(columns={"english": "src", language: "tgt"})

for lang in languages:
  wat_dataset_id = f"DebasishDhal99/{lang}-visual-genome-instruction-set"
  # for split in ["train", ""]
  df1 = load_dataset(wat_dataset_id, split = "test").to_pandas()[["english", lang]]
  df1['tokens'] = df1[lang].apply(lambda x: len(x.split()))
  # print(lang, df1['tokens'].sum())
  print(df1.info())

# Change split to "train", "dev" and "challenge" to get stats for those splits


  