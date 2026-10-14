# tsfm: a small time-series foundation model, trained from scratch

**Question.** How far can a small decoder-only transformer, pretrained from scratch on a
public non-leaking corpus plus synthetic series, get on the GIFT-Eval zero-shot benchmark
compared with classical baselines and a published checkpoint of similar size?

**Status.** Work in progress. Everything below is generated from committed files in
`results/` by `tsfm docs build`; a test fails if this file contains a number that is not
in those files. Sections without results say *pending*.

## Results

Headline comparison (test split, scored once per model, pre-registered):
{{ results/eval/headline.json#summary_md | default:*pending: test scoring happens only after the pre-registration tag (milestone 5).* }}

## Reproduce

```bash
uv sync --extra cpu
uv run pytest
uv run tsfm docs check
```

Training runs on Google Colab with `notebooks/colab_runner.ipynb`.

## Limits

Pending; see the "not established" section of the research note once written.
