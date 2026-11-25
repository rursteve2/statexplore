# StatExplore

**A desktop program for exploring a dataset, searching it for interesting statistical
interactions, and teaching what those results do and don't mean.**

StatExplore runs on Windows, entirely on your own computer: no internet connection, no
uploads, no accounts. You open a data file (Excel, CSV or SPSS). The program describes your
variables, runs standard statistical tests, searches automatically for interaction effects, and
draws graphs with a short plain-English interpretation underneath. It also shows you the exact
statistical code behind every result.

**New in 1.1:** a *Measurement Review* that flags variables whose declared level looks wrong (for
example ordered answer scales stored as "nominal"), a per-analysis **"treat as continuous"** option,
frequency tables, cross-tabulations and categorical plots, odds / risk ratios and risk differences,
ordinal and multinomial outcomes in Discovery and regression, binary / ordinal / multinomial mixed
models and GEE, a longitudinal data check before fitting, survey-weighted analysis, an **R Console**
for your own R code, and an offline **R Package Manager**.

**New in 1.2:** Discovery searches **every combination of interactions** (main effects only, one
interaction, several interactions at once, full factorial models) within six distinct predictors per
model, and reports each tested hypothesis separately (interaction terms, joint interaction tests,
main effects, whole-model tests, each with its own multiple-testing family). Results tables sort
numbers correctly, **highlight** findings with adjusted p < .05 by default (or a stricter rule that also
requires the confidence interval of the same coefficient to exclude its null value), and stay fast with
hundreds of thousands of rows. **Longitudinal discovery** is set up explicitly (participant ID + time)
with random slopes, AR(1) / continuous-time AR(1), marginal (mmrm) models, binary and ordinal mixed
models and GEE, plus a separate comparison of covariance structures. The **treatment panel** now lists
the outcome and every selected predictor with its status, and manual models have adjustable complexity
limits.

> **Important:** StatExplore is built for *exploration and teaching*, not for publishing or making
> decisions. Searching many models will always turn up some "significant" results by chance. The
> program is designed to show you that, so treat its discoveries as ideas to test, not as proof.

---

## Contents
1. [What you need](#1-what-you-need)
2. [Installing and starting StatExplore](#2-installing-and-starting-statexplore)
3. [A first session, step by step](#3-a-first-session-step-by-step)
4. [The sections of the program](#4-the-sections-of-the-program)
5. [How to read the results](#5-how-to-read-the-results)
6. [Five things to always keep in mind](#6-five-things-to-always-keep-in-mind)
7. [Saving, exporting and sharing](#7-saving-exporting-and-sharing)
8. [Troubleshooting and FAQ](#8-troubleshooting-and-faq)
9. [Glossary](#9-glossary)
10. [What is not finished yet](#10-what-is-not-finished-yet)
11. [For developers](#11-for-developers)

---

## 1. What you need

- A Windows 10 or Windows 11 computer.
- About 1 GB of free disk space.
- Your data in one of these formats:
  - **Excel** (`.xlsx`): you choose which sheet.
  - **CSV** (`.csv`): comma- or semicolon-separated text.
  - **SPSS** (`.sav`): variable labels, value labels ("1 = Male"), and missing-value codes are kept.

Your data should have **one column per variable** and **one row per observation**. For data
collected repeatedly from the same people (for example yearly surveys), StatExplore works best
with one row per person per occasion. Section 4 explains how to convert data that has one row per
person with columns like `score_2010`, `score_2012`.

StatExplore uses a free statistics program called **R** behind the scenes. The stand-alone program
folder and the (future) installer include R; when running from the source code you install R once
yourself (see 2B).

---

## 2. Installing and starting StatExplore

### Download it (for most people)
1. Go to the **[Releases page](https://github.com/rursteve2/statexplore/releases/latest)**.
2. Under **Assets**, download **one** of:
   - **`StatExplore-1.3.0-Setup.exe`**: the installer (recommended). Double-click it and click
     through; then start **StatExplore** from the Start menu or the desktop shortcut.
   - **`StatExplore-1.3.0-portable.zip`**: no installation. Right-click ▸ *Extract All…* into, for
     example, your Documents folder, open the extracted `StatExplore` folder and double-click
     **`StatExplore.exe`**.
3. **If Windows says "Windows protected your PC":** click **More info**, then **Run anyway**. This
   appears for any program downloaded from the internet that isn't signed with a paid certificate.

That's all: R and every statistical package are included. You don't need to install Python or R.
Requirements: Windows 10 or 11 (64-bit) and about 1.5 GB of free disk space.

> The green **Code ▸ Download ZIP** button on the main GitHub page downloads the *source code*, not
> the program. Use the Releases page above.

The sections below are for running from source code or building a release yourself.

### 2A. On the computer where StatExplore was developed (easiest)
Open the `statexplore` folder and double-click **`run.bat`**. Everything it needs (Python environment,
R and the R packages) is already set up on that computer.

### 2B. From the GitHub repository, on another Windows computer
This takes about 20 minutes, once.
1. Install **Python 3.11** from python.org (tick *"Add python.exe to PATH"* during setup).
2. Install **R for Windows** (version 4.4 or newer) from cloud.r-project.org.
3. Download the repository: on GitHub click **Code ▸ Download ZIP**, then unzip it (or `git clone` it).
4. Open the unzipped `statexplore` folder, click the address bar, type `cmd` and press Enter.
   In the black window, paste these lines one at a time:
   ```bat
   py -3.11 -m venv .venv
   .venv\Scripts\python -m pip install -e .
   "C:\Program Files\R\R-4.6.2\bin\x64\Rscript.exe" r\install_packages.R
   ```
   (Change `R-4.6.2` to the R version you installed. The last step downloads the R packages once.)
5. From then on, double-click **`run.bat`**.

### 2C. Building a release (for whoever distributes the program)
1. Do 2B, plus `.venv\Scripts\python -m pip install -e ".[dev]"`.
2. Install **Inno Setup 6** (free): `winget install JRSoftware.InnoSetup`.
3. Run `.venv\Scripts\python packaging\build.py`. In about 15 minutes this creates:
   - `dist\StatExplore\`: the stand-alone program (StatExplore.exe with R and all packages inside);
   - `dist\release\StatExplore-<version>-portable.zip` and `dist\release\StatExplore-<version>-Setup.exe`.
4. Check it: `dist\StatExplore\StatExplore.exe --selftest --full` runs one real analysis from every part
   of the program plus all exports (report in `%TEMP%\statexplore_selftest.txt`; exit code 0 = OK).
5. Publish: `gh release create v<version> dist\release\* --notes-file docs\RELEASE_NOTES_v<version>.md`.

### What you should see

- A window with a dark blue **sidebar** on the left listing the sections.
- At the bottom right, the status bar first says **"Starting R…"** and then, after a second or
  so, **"R ready: R version 4.6.2"**. That means the statistics engine is working.
  - If it says **"R not available — see Settings"**, see [Troubleshooting](#8-troubleshooting-and-faq).

---

## 3. A first session, step by step

This walkthrough uses a Health and Retirement Study (HRS) extract, `hrs_least_missing_1000.sav`
(1,000 people, interviewed up to 16 times), which was used during development. HRS data is **not**
included in this repository; you need your own copy. The same steps work for any dataset.

### Step 1: Open your data
- Click **File ▸ Import Dataset…** (or drag the file onto the window).
- For Excel files you'll be asked which sheet; for CSV files, how the columns are separated (leave
  it on "Auto-detect" if unsure).
- The **Dataset Explorer** opens. The **Overview** tab lists every variable with its type,
  missing values, averages and possible outliers.

**Your original file is never changed.** StatExplore works on its own copy.

### Step 2: Check your variables
Open the **Variables** tab. Click a variable to see its label and codes. Check two things:

- **Measurement level.** *Continuous* means a number you can average (age, income, a score).
  *Nominal* means categories with no order (gender, race). *Ordinal* means ordered categories
  (never / sometimes / often). *Binary* means exactly two categories.
  Files sometimes get this wrong. Open the **Measurement Review** tab: it lists every variable whose
  values and labels suggest a different level, with the reason. In the HRS file, the activity
  frequencies (`R_LTACTX`: 1 = every day … 5 = never) and self-rated health (`R_SHLT`: 1 = excellent …
  5 = poor) are stored as *nominal* but are ordered scales, and *years of education* (`RAEDYRS`) is
  stored as nominal but is a number. Nothing changes until you click **Accept suggestion**, **Keep
  original classification** or **Set this level** — and confirm. The file's own level is always kept
  alongside yours. See [4A](#4a-measurement-levels-and-treat-as-continuous).
- **Role.** If the same person appears on several rows, set their ID variable's role to **id**
  (in HRS: `HHIDPN`), and the occasion variable to **time** (in HRS: `WAVE`). This matters: it lets
  StatExplore warn you when rows aren't independent and enables the longitudinal tools.

Click **Apply changes**. Every change is logged and can be undone with **Undo last change**.

> **HRS tip:** `R_CESDM` is the *number of missing* depression items, **not** a depression score.

### Step 3: Create a score from several items (optional)
In the **Derived & Filters** tab, click **New derived variable…**. For example, to build the
8-item CES-D depression score:
1. First reverse the two positive items: operation **Recode**, source `R_WHAPPY`, rows `0 → 1`
   and `1 → 0`, name it `WHAPPY_R`. Do the same for `R_ENLIFE`.
2. Then operation **Sum of items** with the eight items (using the two reversed ones), named `CESD8`.

You can also **add filters**, for example *keep rows where WAVE equals 10* to analyse one wave only.
Filtered-out rows turn grey in the Data tab.

### Step 4: Look at the data
In **Exploratory Data Analysis**, pick variables on the left:
- **Describe (psych)** gives the average, SD, median, skew and more for the chosen variables
  (optionally split by a group).
- **Draw plot** gives a histogram, boxplot, scatterplot, etc. The program suggests a suitable plot.
- **Correlation tests** gives correlations with ordinary and multiple-testing-corrected p-values.

### Step 5: Run a standard analysis
**Analysis Builder** contains t-tests, correlations, ANOVA, ANCOVA, linear, hierarchical and
logistic regression. Choose an analysis on the left, put variables into the boxes, then click
**Run analysis**. Results show tables, graphs and any warnings.

### Step 6: Search for interactions (the main feature)
In **Automated Model Discovery**:
1. Choose the **dependent variable** (the outcome, e.g. `CESD8`) and confirm its **model family**.
2. Tick the **candidate predictors** (up to 39).
3. Under **Study structure**, keep *Independent observations* for one row per person. If your data has
   several rows per person, choose **Repeated measures / longitudinal**, then the **participant ID** and
   the **time** variable; the preflight shows rows, participants, waves and observed outcomes by wave.
4. Choose a **search strategy**. The default, *All eligible interaction combinations*, fits every
   hierarchical combination of interactions on every subset of predictors (for three predictors:
   main effects only, each two-way interaction, each pair of interactions and all three; with
   three-way interactions enabled, also `X1 * X2 * X3`). *Single focal interaction*, *Full factorial*,
   *Main effects only* and (longitudinal) *Time × each predictor* are the alternatives.
5. Click **Calculate search space** to see how many models and model fits the search needs, then
   **Start search**.

The **maximum of six predictors** counts *distinct variables*: `Y ~ X1 + X2 + X3 + X1:X2 + X2:X3` uses
3 of 6, has 5 model terms and 6 estimated coefficients. The tables show all three numbers.

Results appear in five tabs:
- **Summary:** how many models and hypotheses were tested, by hypothesis type, how many look
  significant, and how many you'd expect by chance alone.
- **Discovery significance ranking:** one row per tested hypothesis (what was tested, against which
  reference model, with which test), sorted by p-value.
- **Statistical support ranking:** better-supported findings first (clean checks and significant
  after correction).
- **Model comparison:** fitted models with AIC / BIC / R² — only compared when they use the same rows.
- **All attempts:** every model, including ones that couldn't be fitted, with the reason.

The **Highlight findings** bar marks rows with adjusted p < .05 (change α, switch to raw p, the strict
same-estimand-CI rule, or off; *Show highlighted only*; counts of highlighted rows). Highlighting never
refits anything. Use **Show** to look at one hypothesis type or model type at a time; screening warnings
(convergence, singular fits, sparse cells) are flagged separately from significance.

Select a row to **Open full analysis**, **Plot in Teaching Studio** (longitudinal models open in **Mixed
& Longitudinal Models**, refitted with the same formula and structure), **Add to comparison** or
**Compare with main-effects model**.

### Step 7: Understand an interaction
**Teaching Studio** draws the interaction from the fitted model, with lines for each group or
level and shaded confidence bands. Underneath is a short **Interpretation** in plain English.
Change the x-axis, grouping, panels, labels and format (PNG, SVG, PDF) on the left, and choose the
values at which the other predictors are held (by default their average, or the reference category).

### Step 8: Check whether a finding holds up
In **Validation & Uncertainty**:
- **Resampling stability:** reruns the search on many random subsets of people and shows how often
  each finding reappears.
- **Full-procedure CV:** repeats the *whole search* on part of the data and tests the winner on the
  rest.
- **Holdout:** set aside a portion of the data **before** exploring, and test your final idea on it
  once at the end.

### Step 9: Save and export
**File ▸ Save Project** saves everything (data copy, settings, results, graphs) in one `.sxp` file.
**Reports** exports to Excel, Word, PDF and R scripts.

---

## 4. The sections of the program

| Section | What it's for |
|---|---|
| **Dashboard** | Overview of the open project and recent analyses. |
| **Dataset Explorer** | **Overview** of every variable; **Data** spreadsheet view (tick *Show value labels* to see "Male" instead of 1; red = a declared missing code); **Variables** to set measurement level, role, reference group and display name; **Derived & Filters**; **Missing data** (how much is missing, common patterns, how many rows each model can use). |
| **Exploratory Data Analysis** | Descriptive statistics (psych) for continuous variables; **Frequencies** (counts, %, valid %, missing, mode, number of categories; cumulative % and median category for ordinal) for categorical ones — no means of category codes; **Cross-tabulation** (row / column / total %, expected counts, chi-square with Fisher advice); bar, percentage, horizontal, clustered, stacked, 100 % stacked, mosaic and ordinal-distribution plots; correlation tests (Spearman ranks ordinal categories; Pearson needs "treat as continuous"). Every list has search, *Check visible*, *Clear visible*, *Check all*, *Clear all* and a count. |
| **Analysis Builder** | Standard tests and regressions with effect sizes, confidence intervals, assumption checks and diagnostic plots. **Categorical Association**: chi-square / Fisher's exact (r × c, Cramér's V) and the **2 × 2 table** (you choose the event and reference outcome categories and the exposed and comparison groups; the table is shown before anything is computed; odds ratio, risk ratio, risk difference with labelled CI methods, phi). **Ordinal** and **multinomial** logistic regression, a **Likert sensitivity** comparison (Gaussian on the codes vs ordinal), **Mediation** and **Scale reliability**. **Copy tables** pastes the results into Word or Excel. |
| **Automated Model Discovery** | Systematic search over main effects and interactions (see Step 6). You must confirm the **outcome model family** (Gaussian, binary logistic, ordinal logistic or multinomial logistic; a suggestion is shown); every hypothesis is tested with the family's own nested-model test. Longitudinal searches lock one random-effect / covariance structure (workflow A) or compare structures with the fixed part locked (workflow B, REML). Modes: *Exhaustive* tests everything in the defined space; *First significant* / *Target number* stop early (labelled incomplete); *Budget-limited* stops after a set number of models or minutes. Profiles Fast / Balanced / Thorough set sensible budgets. You can **Cancel** and later **Resume**. |
| **Model Comparison** | Put models side by side (AIC, BIC, R²). Models that contain one another get a formal test. **Edit a copy** changes a model without losing the original. |
| **Validation & Uncertainty** | Holdout, cross-validation, stability, full-procedure CV, prediction intervals (see Step 8). |
| **Mixed & Longitudinal Models** | For repeated measurements. **Detect structure** suggests the ID and time variables; **Wide → long** converts one-row-per-person data. The **preflight** panel shows, before fitting, how many records, participants and observed values each time point has (e.g. `R_LTACTX` is not asked in HRS waves 1–6, so the analysis range is waves 7–16). Outcome families: **Gaussian** (random intercepts or slopes, AR(1), unstructured; inference by Satterthwaite, Kenward–Roger or likelihood-ratio tests), **binary logistic** (glmer: odds ratios, conditional *and* population-average predicted probabilities), **ordinal** (clmm) and **multinomial** (mblogit, an approximate PQL method — limitations are shown). **GEE** fits population-average models with robust standard errors. |
| **Survey Analysis** | Declare the sampling weight, PSUs, strata (and optionally a finite population correction or replicate weights); then weighted means, proportions and frequency tables, survey-adjusted cross-tabs (Rao–Scott), and survey-weighted linear / logistic regression. StatExplore never guesses design variables (in HRS extracts, `R_WEIGHT` is body weight, not a survey weight). |
| **Teaching Studio** | Interaction graphs with interpretations (see Step 7). **Find teaching examples** lists analyses already run that illustrate a point (significant after adjustment, significant only before adjustment, unstable estimates, assumption problems, Gaussian vs ordinal disagreement, GLMM vs GEE). |
| **R Console** | Run your own R code on a copy of the data (`my_data`). Each run starts a clean R session unless you switch on *Persistent session*. Output, warnings, errors and plots are shown separately; scripts can be opened and saved. Changes to `my_data` reach the project only through **Import resulting dataset**, which shows what will change and saves a recoverable snapshot first. **Your code runs with your Windows permissions: the console is not a sandbox.** **Package Manager** installs R packages from files on your computer (never downloads), into a separate library that cannot replace StatExplore's own packages. |
| **Reports** | Export selected analyses and searches. |
| **Settings** | Where R is; **Test R connection**; **R Package Manager**. |

Almost every result has a **View Code** button. It shows the *exact* R code that produced the result,
which you can copy into R or RStudio.

### 4A. Measurement levels and "treat as continuous"

StatExplore keeps three things apart:

1. **Source level** — what the file declares (e.g. SPSS *nominal*). Never changed.
2. **Project level** — what you approved in the Measurement Review or the Variables tab. Logged and undoable.
3. **Treatment for one analysis** — how a variable is used in *this* analysis only. The treatment panel
   always lists the **outcome** (directly under the outcome family) and **every selected predictor**:
   eligible numeric ordinal / nominal variables get a checkbox **Treat … as continuous for this
   analysis**; continuous ones say *Already continuous — no override needed*; others say *Unavailable*
   with the reason (text categories, constant, identifier). *All eligible dataset variables* lets you
   preconfigure treatments, which apply only when that variable is in the model. Ticking it asks you to acknowledge that this assumes the numeric codes can be
   read as quantities (and, for a nominal variable, to confirm that the codes really are numbers).
   The analysis then uses the original codes (1–5 for `R_LTACTX`; higher = *less* frequent activity),
   never an arbitrary factor index. The treatment is shown with the result, in the R code, in exports
   and in the saved project; the variable's levels in the project do not change.

---

## 5. How to read the results

- **p-value:** how surprising the result would be if there were really no effect. Small (below .05)
  means "surprising". It does **not** tell you how large or important an effect is.
- **Adjusted p-value (BH / Holm):** a p-value corrected for running many tests. In a search,
  **use these**. *BH* controls the share of false discoveries; *Holm* is stricter.
- **Effect size:** how big the effect is. In discovery, **ΔR²** is the share of the outcome's
  variation the interaction explains on its own: 0.01 is 1%, which is small.
- **Confidence interval (CI):** a range of plausible values for an estimate. Wide intervals mean
  uncertainty.
- **Status:** *ok*; *warning* (read the "Check:" messages); *invalid* (the model couldn't be
  estimated, with the reason); *failed* (an error, shown in red).
- **"Check:" messages** flag real problems: rows that aren't independent, too few cases in a group,
  a model that didn't converge, perfect prediction in logistic models ("separation").
- **n used / excluded:** how many rows went into the analysis. Rows with missing values on the
  analysis variables are left out, and the program tells you how many.
- **Interpretation box (Teaching Studio):** three to six sentences generated from fixed templates
  using the actual numbers. It always ends with a limitation.

### 5A. Reading results correctly (new in 1.3)

- **Hypothesis families.** A search can test several kinds of hypotheses: individual interaction
  terms, joint tests of interaction terms, main-effect terms, whole-model tests and (longitudinal
  workflow B) covariance-structure comparisons. Each kind is its own multiple-testing family: BH and
  Holm adjustment is applied **within** a family, and the summary counts each family separately. A
  significant whole-model or main-effect test is **not** a significant interaction; "0 significant
  interactions" means none of the interaction tests passed, whatever the other families show.
- **Strict highlighting** needs a confidence interval for the *same* coefficient the p-value tests.
  Joint tests and covariance-structure comparisons have none, so they are never strictly highlighted:
  zero strict highlights does not mean zero significant tests.
- **ML vs REML.** Workflow A (fixed-effect discovery) compares fixed effects with **ML** fits.
  Workflow B (covariance-structure comparison) keeps the fixed part locked and uses **REML**. The
  *Estimation* column says which; never compare REML likelihoods of models with different fixed
  effects. R's own BIC for REML `nlme` fits uses log(n − p); StatExplore's tables use log(n) for every
  engine, so exported scripts can show a BIC that differs in the second decimal.
- **Parameter counts.** *Fixed coefficients* (with ordinal thresholds) are what the search's
  parameter limit counts; *Covariance parameters* are random-effect and residual variance/correlation
  parameters; *Likelihood parameters* is their total, the k in AIC = −2 logLik + 2k.
- **Time index.** *AR(1)* uses the **observed-occasion index**: the rank of the observed time values.
  If a wave is missing for everyone (for example Wave 15), the waves on either side are treated as
  adjacent. *Continuous-time AR(1) (CAR1)* uses the **actual time differences**. They estimate
  different things; a better AIC for one does not make its time definition the scientifically right
  one. The preflight lists the observed time points and any wave absent inside the range.
- **Model identity.** Opening a discovery model elsewhere (Full analysis, Mixed, Model Comparison)
  refits exactly that model on the same rows and then checks n, log-likelihood and AIC against the
  search. The result says "Model identity verified", or the check fails visibly. A destination that
  cannot represent a model says so; it never substitutes a different model.
- **CSV typing.** Empty and whitespace-only cells are read as missing values (never 0), and numbers
  with surrounding spaces as numbers. A numeric column with only a few integer codes (≤ 9 distinct
  values, no labels) is read as categories and flagged in Measurement Review. Choose continuous or
  ordinal there if the codes are quantities or ordered. Zero-padded codes such as `001` stay text.
  `.sav` files keep their SPSS measurement levels, labels and user-missing codes.
- **Exported scripts.** *View candidate code ▸ Save script + data…* writes `analysis.R` and
  `analysis_data.csv` (only the columns the analysis uses). The script finds the data next to itself,
  so `source("analysis.R")` works from any folder. Workflow B exports also recompute the reported
  covariance-structure comparisons.

---

## 6. Five things to always keep in mind

1. **More tests means more false alarms.** With 100 tests at p < .05, about 5 look significant
   even when nothing is real. The discovery summary prints how many to expect by chance.
2. **"Significant" is not "reliable".** Use the support ranking, the stability check and a holdout
   before believing a discovery.
3. **Repeated measurements are not independent.** If a person appears on several rows, set their
   ID role. StatExplore will then warn you, and you can use mixed models.
4. **Association is not causation.** These are observational data.
5. **Effect sizes from a search are inflated.** The winner of a search looks better than it really is.

---

## 7. Saving, exporting and sharing

- **Project file (`.sxp`):** File ▸ Save Project. One file holds a copy of the data, your variable
  settings, derived variables, filters, holdout, all results, graphs, searches and model sets.
  Open it later with File ▸ Open Project. The original data file isn't needed again.
- **Reports page:** tick analyses and searches, choose formats, pick a folder:
  - **Excel:** every table, plus summaries and code.
  - **Word / PDF:** a readable report with interpretations, graphs and tables.
  - **R script (+ data.csv):** reproduces each analysis outside StatExplore. Run it in R or RStudio
    from the export folder.
  - **Python script:** reruns the R script and loads the result.
  - Plus the analysis history and a log of every variable change.
- **Privacy:** nothing leaves your computer. Note that project files and exports contain your data;
  share them only as you would the data itself.

---

## 8. Troubleshooting and FAQ

**The status bar says "R not available — see Settings".**
StatExplore couldn't find R. If you used the installer, reinstall it. If you run from the project
folder, install R for Windows, then in **Settings** click **Browse…**, select `Rscript.exe`
(usually `C:\Program Files\R\R-4.6.2\bin\x64\Rscript.exe`), then **Save R path** and
**Test R connection**. The test also lists any missing R packages.

**"Some combinations of factor levels have no observations."**
A group combination has no data (for example no women in one category). Merge small categories with
a derived **Recode** variable, or leave that variable out.

**"Check: … rows come from only … participants."**
Your data has repeated rows per person. Filter to one occasion, or use **Mixed & Longitudinal
Models**, or choose **Repeated measures** in Discovery.

**A discovery search says it will take very long.**
Reduce the number of candidate predictors, keep *Max predictors per model* at 3, use the default
*Focal* covariate setting, or choose the *Fast* profile. You can always Cancel and Resume.

**"… has the same value in every row" / "… looks like an identifier" / "… is the participant ID".**
StatExplore stops models that cannot work before running them. Remove the variable named in the
message: a constant variable can't explain anything, and ID numbers (or variables with hundreds of
categories) aren't predictors.

**Many models are "invalid".**
Usually a categorical variable has many categories (each one adds parameters), or some groups are
empty. Check measurement levels: a number stored as *nominal* (like years of education in HRS)
should be *continuous*.

**The flexplot backend is greyed out.**
flexplot is an optional plotting add-on that must be installed separately (see the developer
guide). ggplot2 and base R graphs work without it.

**Where is the log file?** `C:\Users\<you>\AppData\Local\StatExplore\statexplore.log`.

**Can I break my data?** No. Imports are read-only, and every change is stored as a separate,
undoable setting.

---

## 9. Glossary

| Term | Meaning |
|---|---|
| **Dependent variable / outcome** | What you're trying to explain (e.g. depression score). |
| **Predictor** | A variable that may explain the outcome. |
| **Interaction** | When the effect of one predictor depends on another (e.g. exercise relates to depression differently for men and women). Written *A × B*. |
| **Covariate** | A variable included to adjust for it. |
| **Reference level** | The category other categories are compared with in regression. |
| **Continuous / nominal / ordinal** | Number / unordered categories / ordered categories. |
| **Effect size** | How big an effect is, regardless of sample size. |
| **Confidence interval** | Plausible range for an estimate. |
| **Prediction interval** | Plausible range for **one new person's** outcome (much wider than a CI). |
| **Multiple-testing correction** | Adjusting p-values because many tests were run (BH, Holm). |
| **Holdout** | Data set aside and only used at the end to test a finding. |
| **Cross-validation** | Repeatedly fitting on part of the data and testing on the rest. |
| **Mixed model** | A model for repeated measurements that allows each person their own level (random intercept) and change (random slope). |
| **AR(1)** | Errors from neighbouring occasions are correlated. |
| **Singular fit** | A mixed model with a component estimated as zero: simplify it. |
| **Separation** | In logistic regression, a predictor perfectly predicts the outcome, so its estimates are unreliable. |
| **Johnson-Neyman interval** | The range of the moderator where the effect of x is significant. |

---

## 10. What is not finished yet

- **Testing on other computers:** releases are built and checked on the development computer. They
  haven't yet been tried on a freshly installed Windows 10/11 machine; please report any problem via
  GitHub **Issues**.
- **flexplot** plotting add-on: optional, needs a separate install (developer guide).
- Games-Howell post hoc for Welch ANOVA, omega / factor analysis, robust standard errors outside GEE.
- **Multinomial mixed models** use penalized quasi-likelihood (PQL), an approximation that can
  underestimate random-effect variances; results say so. Likelihood-ratio tests are not available for them.
- Repeated **multinomial** outcomes are not offered in Discovery (the PQL approximation has no valid
  likelihood-ratio test); use **Mixed & Longitudinal Models**. Resampling stability and full-procedure CV
  are available for single-level Gaussian and binary searches only.
- Longitudinal discovery candidates are refitted in **Mixed & Longitudinal Models** for graphs (labelled as
  a refit); mmrm compound-symmetry / AR(1) structures are fitted in Discovery but not offered there.
- Four-way and higher interactions are not searched (maximum interaction order 3). Very large searches
  (hundreds of thousands of hypotheses) take a few seconds to display.
- Confidence intervals of models selected from a search are not corrected for the selection.
- R packages installed through the Package Manager are specific to the R version (4.6); source
  packages with compiled code need Rtools, which StatExplore does not include.
- Multiple imputation of missing values isn't included. Missing data is handled by complete cases,
  or by all available occasions in mixed models.
- Resuming a search does not retry candidates that timed out; start a new search with a longer limit.
- Exports replace numeric participant IDs by codes 1, 2, …; text IDs are kept as they are.
- No participant × wave coverage chart yet (per-wave availability tables are shown instead). Form settings are
  saved with the project (not autosaved between saves).
- Clustered (three-level) Discovery is Gaussian only; clustered GEE, marginal (mmrm), binary and ordinal mixed
  models are refused in Discovery with a reason. Crossed or multiple-membership designs are not supported.
- Results from before 1.3.2 carry no identity fingerprints: their refits are reported as unverifiable.

**New in 1.3.2:** Automated Model Discovery handles participants nested in one higher-level cluster (school,
hospital, …) in both workflows — fixed-effect discovery with the hierarchy locked, and the full random intercept /
uncorrelated slope / correlated slope × independent / AR(1) / CAR(1) structure table, with or without the cluster —
plus organization-, participant- and future-visit validation. Model identity is now checked with fingerprints of the
rows, outcome coding and design (equal fit statistics never count as the same model); historical results are refitted
on their own data view or run as clearly labelled new analyses; likelihood-ratio tests use the reference
distribution of the tested hypothesis; diagnostics flag near-boundary and optimizer problems per model; R² is a
direct variance decomposition. See `docs/RELEASE_NOTES_v1.3.2.md`.

**New in 1.3.1:** the proportional-odds check reports a p-value or a specific reason for every term; BIC is
R's native `BIC()` everywhere; exported scripts check their own results, data file and packages; and
Measurement Review, the variable lists, the scroll bars and the result tables were made easier to use (see
`docs/RELEASE_NOTES_v1.3.1.md`).

---

## 11. For developers

How each requirement of the original specification is met: **[docs/spec-compliance.md](docs/spec-compliance.md)**;
for the 1.1 upgrade: **[docs/spec-compliance-v1.1.md](docs/spec-compliance-v1.1.md)**; for 1.2:
**[docs/spec-compliance-v1.2.md](docs/spec-compliance-v1.2.md)** (with a feature matrix); for 1.3 (defect
traceability, test evidence, limitations): **[docs/spec-compliance-v1.3.md](docs/spec-compliance-v1.3.md)**.
Release notes: [docs/RELEASE_NOTES_v1.3.0.md](docs/RELEASE_NOTES_v1.3.0.md), [1.2.0](docs/RELEASE_NOTES_v1.2.0.md),
[1.1.0](docs/RELEASE_NOTES_v1.1.0.md).
The discovery grammar is in [docs/formula-grammar.md](docs/formula-grammar.md).

Building, testing, packaging and the architecture are described in
**[docs/DEVELOPER.md](docs/DEVELOPER.md)**. The development plan and per-phase reviews are in
[tasks/todo.md](tasks/todo.md).

---

## License

StatExplore is released under the **MIT License** (see [LICENSE](LICENSE)). The downloadable
program also contains R, R packages, Python and Qt, each under its own license. See
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
