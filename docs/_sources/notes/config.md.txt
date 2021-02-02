## Configuration

Tasks that utilize **Transformer** architecture take the configuration as shown below

<br>

### Transformer

- Set the value of **Transformer** task model in `tasks/utils/config.py`
- the model value can be set as follows:

```python
# Format
@dataclass
class TransformerConfig:
    src_dict: Union[str, None]  # source dictionary
    tgt_dict: Union[str, None]  # target dictionary
    src_tok: Union[str, None]   # source tokenizer
    tgt_tok: Union[str, None]   # target tokenizer

# Usecase
"transformer.base.ko.pg":
    TransformerConfig(
        "dict.transformer.base.ko.mt",
        "dict.transformer.base.ko.mt",
        "bpe8k.ko",
        None,
    ),
```

- The model load can be done in the following way:

```python
# Pass the model name to download, and then get the path
load_dict = download_or_load(f"transformer/{self._n_model}", self._lang)

# Use the path information to load the model
model = TransformerModel.from_pretrained(
  model_name_or_path=load_dict.path,
  checkpoint_file=f"{self._n_model}.pt",
  data_name_or_path=load_dict.dict_path,
  source_lang = load_dict.src_dict,
  target_lang = load_dict.tgt_dict,
)

# Load the tokenizer, if necessary
tokenizer = CustomTokenizer.from_file(
    vocab_filename=f"{load_dict.src_tok}/vocab.json",
    merges_filename=f"{load_dict.src_tok}/merges.txt",
)
```
