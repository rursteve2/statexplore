# Capability matrix (v1.3.3)

What each model family supports in Full Analysis (Mixed & Longitudinal Models), its comparisons, AMD and AMD
validation. "Refused" always comes with the reason in the application; nothing is silently replaced by a simpler model.

| Family / engine | Full Analysis exact refit + lineage | Fitted-vs-requested check | Time-form comparison | Random-effects comparison | Residual / working-correlation comparison | AMD workflows | AMD validation |
|---|---|---|---|---|---|---|---|
| Gaussian lme4 (independent residuals) | yes | yes | ML; keeps random part, cluster, terms; nested LR | REML, boundary-aware LR (RI→DIAG 0:1, RI→RS 1:2, DIAG→RS χ²₁ only with interior variances) | via nlme REML (AR(1) χ²₁, CAR(1) boundary, AR(1) vs CAR(1) AIC/BIC only) | A and B, with / without cluster (18 × 2) | yes, all targets |
| Gaussian nlme (AR(1) / CAR(1)) | yes (corARMA(1,0) accepted as nlme's AR(1) with index gaps) | yes | ML; keeps AR(1) / CAR(1) and cluster | REML | REML | A and B | yes, all targets |
| Gaussian mmrm (unstructured) | yes | yes (class, terms) | refused (needs categorical time) | refused (no random effects) | refused | A (categorical time) | refused: no new-participant prediction implemented |
| Binary GLMM (glmer) | yes | yes | ML (Laplace / AGQ) nested LR | Laplace LR where nested | refused (no residual structure; use GEE AR(1)) | A, participant level | refused: marginal prediction needs integration over random effects |
| Ordinal CLMM (clmm) | yes | yes (class, terms, grouping) | ML (Laplace) nested LR | RI vs RS | refused | A, participant level | refused (same reason) |
| Multinomial mixed (mblogit, PQL) | yes | yes (class, terms) | refused (no likelihood) | refused | refused | A | refused |
| GEE (geepack) | yes | yes (class, terms, working correlation) | robust Wald of the quadratic term | refused (no random effects) | QIC / QICu / CIC only, no LR / AIC | A | yes (marginal mean), participant target |
| Single-level linear / logistic | (Analysis Builder) | — | — | — | — | all strategies | rows target (Gaussian: RMSE / MAE / R²; binary: log loss / Brier / AUC) |

Clustered (three-level) designs: Gaussian only, one higher-level cluster with participants nested; crossed or
multiple-membership designs are refused. Weighted multilevel / longitudinal models are not supported (survey weights
apply to independent-observation analyses only).
