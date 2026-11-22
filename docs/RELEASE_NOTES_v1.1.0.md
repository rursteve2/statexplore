# StatExplore 1.1.0

An upgrade of StatExplore 1.0 focused on categorical and ordinal data, fewer unnecessary restrictions,
more model families, survey analysis and your own R code. Everything still runs offline; R and every
statistical package are included. Projects saved with 1.0 open unchanged (a backup of the old file is
kept the first time you save it in the 1.1 format).

## Download (choose one)
- **`StatExplore-1.1.0-Setup.exe`**: installer (recommended).
- **`StatExplore-1.1.0-portable.zip`**: no installation; unzip and run `StatExplore.exe`.

Windows may say "Windows protected your PC" (the program is not code-signed): **More info ▸ Run anyway**.
Requirements: Windows 10 or 11 (64-bit), about 1.5 GB of free disk space.

Checksums (SHA-256): Setup `8a634fbee48d73fa1c8f6c15126a062f230ef9f6e4ba8ea68f929b10950d55c9` · portable zip `02ad56583f64a7fd8a42273efebbf03f6b53974941e7bf057e9c9ae73cedf3bc`.

## What's new

**Measurement levels**
- *Measurement Review* (Dataset Explorer): flags variables whose declared level does not match their
  values and labels — e.g. ordered answer scales stored as nominal — with the reason. Nothing changes
  without your explicit, confirmed approval; the file's own level is always kept.
- *Treat as continuous for this analysis*: use a numeric categorical or ordinal variable's codes as
  numbers in one analysis (after acknowledging what that assumes), without changing the project.
  Shown with the result, in the R code, in exports and in the saved project.

**Exploration**: frequency tables (no means of category codes), cross-tabulations with expected counts
and test advice, eight categorical plot types, Spearman correlations for ordinal variables, and a
standard variable selector (search, check/clear visible, check/clear all, selected count) on every page.

**Analysis Builder**: Categorical Association (chi-square, Fisher's exact, Cramér's V, phi) and 2 × 2
tables with odds ratio, risk ratio and risk difference — you choose event, reference, exposed and
comparison categories and see the table first. Ordinal and multinomial logistic regression, a Likert
sensitivity comparison (Gaussian vs ordinal), event choice in logistic regression, brief interpretations
and *Copy tables*.

**Automated Model Discovery**: confirmed outcome family — Gaussian, binary, ordinal or multinomial —
with family-specific likelihood-ratio interaction tests and the outcome coding recorded with each search.

**Mixed & Longitudinal Models**: binary (glmer), ordinal (clmm) and multinomial (mblogit, approximate)
mixed models; GEE; Satterthwaite / Kenward–Roger / likelihood-ratio inference for Gaussian models;
a preflight showing availability by time point (no zero-filling of unobserved waves).

**Survey Analysis**: weights, PSUs, strata, FPC or replicate weights; weighted means, proportions,
tables, Rao–Scott cross-tabs, survey-weighted regression.

**R Console and Package Manager**: run your own R code on a copy of the data in a separate R session
(clean per run, or persistent), with plots; import changes only on request, after a snapshot. Install R
packages from local files into your own library (never downloaded; StatExplore's packages are protected).
*Custom R code runs with your Windows permissions: it is not a sandbox.*

**Teaching**: *Find teaching examples* lists completed analyses that illustrate common points.

## Known limitations
See README §10. In short: multinomial mixed models are approximate (PQL); repeated-measures discovery is
Gaussian only; resampling stability is not yet offered for ordinal/multinomial searches; source packages
with compiled code need Rtools; not yet tested on a freshly installed Windows machine.
