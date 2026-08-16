# Data

This project does not vendor raw dataset files in the repository.

Both datasets are streamed on demand through Hugging Face `datasets` the first
time `src/data.py` is invoked (e.g. via `src/train.py`):

| Dataset  | HF identifier      | Task                     | Classes |
| -------- | ------------------- | ------------------------ | ------: |
| SST-2    | `glue` / `sst2`      | Sentiment classification |       2 |
| AG News  | `ag_news`            | Topic classification     |       4 |

`datasets` caches downloads under the Hugging Face cache directory
(`~/.cache/huggingface/datasets` by default, or `$HF_HOME` if set) rather than
inside this project directory, so nothing under `data/` needs to be committed
or manually managed.
