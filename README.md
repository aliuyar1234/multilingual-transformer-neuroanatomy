# Inside the Multilingual Transformer: Computational Neuroanatomy of Shared and Language-Specific Brain Alignment

[![DOI](https://zenodo.org/badge/1187629245.svg)](https://doi.org/10.5281/zenodo.19185601)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-B31B1B?style=flat-square&logo=adobeacrobatreader&logoColor=white)](paper/attention-side-transformer-brain-alignment.pdf)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/aliuyar1234/multilingual-transformer-neuroanatomy)
[![Citation](https://img.shields.io/badge/Citation-CFF-0A7F5A?style=flat-square)](CITATION.cff)
[![Scope](https://img.shields.io/badge/Scope-ROI--First%20Mechanistic%20Study-5B4B8A?style=flat-square)](#scope)

Ali Uyar
Independent Researcher

**Paper title:** *Inside the Multilingual Transformer: Computational Neuroanatomy of Shared and Language-Specific Brain Alignment*

This repository accompanies a mechanistic multilingual neuroscience study rather than a model-benchmark comparison. Using the Le Petit Prince multilingual fMRI corpus (LPPC) in English, French, and Chinese, it asks which internal computations inside XLM-R-base and the encoder of NLLB-200-distilled-600M generate the shared-over-specific brain alignment effect observed in prior work, and why that effect extends into auditory and superior temporal cortex rather than sitting cleanly in semantic ROIs.

## Abstract

Most brain-LLM alignment work uses returned hidden states and is effectively monolingual, leaving unclear which internal computations carry cross-lingually shared explanatory signal. A prior multilingual LPPC study showed that a leave-target-out `SHARED` representation outperformed a target-language `SPECIFIC` residual in semantic ROIs, but auditory and superior temporal regions also showed robust shared advantages. This paper asks where inside multilingual transformers that shared advantage is generated. Using English, French, and Chinese listeners from LPPC, we extracted intermediate block states from XLM-R-base and the encoder of NLLB-200-distilled-600M, decomposed each state into shared and language-specific components, and evaluated them with cross-validated ROI encoding models supplemented by token-level attribution and deletion validation. The originally planned semantic FFN-preference headline was not supported in the confirmatory primary table; instead, auditory attention-side preference was positive in most model-language cells and Holm-significant in 4 of 6. Deletion validation was uniformly positive, with top-attributed token removal exceeding matched-random removal in every released row. These results argue for a state-level, attention-side mechanistic account of multilingual brain alignment rather than a clean semantic-FFN story.

## Main Finding

The originally planned headline (`H1_semantic_ffn_preference`) did not survive. All six confirmatory `model x language` H1 rows came out negative, so the paper does not support an FFN-semantic mechanism for the `SHARED > SPECIFIC` effect. The attention-side preference in auditory/STG regions (`H2_auditory_attention_preference`), by contrast, is positive in most cells and Holm-significant in 4 of 6:

| Model | Language | H2 effect | p_holm        |
| ----- | -------- | --------- | ------------- |
| xlmr  | EN       | 0.0066    | **0.0012**    |
| xlmr  | FR       | 0.0038    | **0.0072**    |
| xlmr  | ZH       | 0.0073    | **0.0022**    |
| nllb  | EN       | -0.0027   | 1.0           |
| nllb  | FR       | 0.0002    | 1.0           |
| nllb  | ZH       | 0.0048    | **0.0050**    |

Deletion validation is the cleanest confirmatory surface in the release. Across every model-language-ROI-family row tested, removing top-attributed tokens degrades encoding more than removing matched-random tokens. Representative deletion-validation differences include XLM-R English semantic at k=2 (`diff = 0.0494`, CI excludes zero) and NLLB Chinese auditory at k=2 (`diff = 0.0380`).

The paper therefore lands on a narrower but mechanistically clearer story: multilingual brain alignment is not explained by an FFN-side semantic preference; it reflects attention-side computations that interact with how tokens carry alignment-relevant signal mass.

## Contributions

1. A state-level multilingual factorization that extracts internal transformer block states rather than relying only on returned hidden states, and decomposes each state into leave-target-out `SHARED` and orthogonalized `SPECIFIC` components.
2. Confirmatory `SHARED > SPECIFIC` primary tests across English, French, and Chinese with two multilingual model families (XLM-R-base and the encoder of NLLB-200-distilled-600M) and Holm correction across the full `model x language x hypothesis` grid.
3. Token-level attribution of multilingual alignment-relevant signal mass, broken down by token class, tied directly to representative ROIs.
4. Deletion validation showing that top-attributed token removal degrades encoding more than matched-random removal in every released row, supporting a causal rather than merely correlational reading of the attention-side account.
5. A fully paper-facing public release surface: frozen manifests, executable analysis code, paper-facing figures and tables, regression tests, and the compiled manuscript PDF.

## Scope

This release is intentionally narrow.

- One public naturalistic fMRI corpus: LPPC (`ds003643`), English/French/Chinese cohorts only
- Two multilingual model families: `FacebookAI/xlm-roberta-base` and the encoder of `facebook/nllb-200-distilled-600M`
- Anatomical ROI-first analysis on a Harvard-Oxford atlas resampled into LPPC derivative space
- Primary claims live at the ROI-family level; the cortex-wide panel is an ROI-projected visualization, not a voxelwise inferential map
- Sentence-span analysis rather than long-context comprehension modeling
- Correlational encoding plus attribution-guided deletion; no intervention in the brain and no training of the transformers themselves

The contribution is mechanistic narrowing, not breadth. The released public surface supports an attention-side account of multilingual brain alignment and does not support the FFN-semantic framing that the project originally planned around.

## Paper

- Compiled PDF: [`paper/attention-side-transformer-brain-alignment.pdf`](paper/attention-side-transformer-brain-alignment.pdf)
- Machine-readable citation: [`CITATION.cff`](CITATION.cff)
- Primary confirmatory stats: [`outputs/tables/table03_primary_confirmatory_stats.csv`](outputs/tables/table03_primary_confirmatory_stats.csv)
- Representative ROI summaries: [`outputs/tables/table04_representative_roi_summaries.csv`](outputs/tables/table04_representative_roi_summaries.csv)
- Token-class attribution: [`outputs/tables/table05_token_class_attribution.csv`](outputs/tables/table05_token_class_attribution.csv)
- Deletion validation: [`outputs/tables/table06_deletion_validation.csv`](outputs/tables/table06_deletion_validation.csv)

## Repository Layout

- [`src/`](src/) — analysis code for manifests, state extraction, features, encoding, statistics, attribution, and figure generation
- [`conf/`](conf/) — sanitized public configuration for the released analysis surface
- [`data/manifests/`](data/manifests/) — frozen structural manifests for the sample, runs, ROIs, sentence spans, and multilingual triplets
- [`outputs/tables/`](outputs/tables/) — paper-facing CSV tables
- [`outputs/figures/`](outputs/figures/) — paper-facing PNG and PDF figures
- [`paper/`](paper/) — compiled paper PDF
- [`tests/`](tests/) — lightweight regression tests for public code paths
- [`templates/`](templates/) — public-safe configuration templates
- [`docs/`](docs/) — public methods, runbook, and scientific-decision notes

## Reproducibility

- [`REPRODUCE.md`](REPRODUCE.md) — reproduction and lightweight validation guidance
- [`DATA_ACCESS.md`](DATA_ACCESS.md) — scope and boundary document for included versus archived assets
- [`docs/RUNBOOK.md`](docs/RUNBOOK.md) — operational runbook for the released analysis surface
- [`docs/SCIENTIFIC_DECISIONS.md`](docs/SCIENTIFIC_DECISIONS.md) — rationale for key design choices

## Citation

```bibtex
@unpublished{uyar2026multilingualtransformer,
  author = {Uyar, Ali},
  title  = {Inside the Multilingual Transformer: Computational Neuroanatomy of Shared and Language-Specific Brain Alignment},
  year   = {2026},
  doi    = {10.5281/zenodo.19185601},
  note   = {Independent research}
}
```
