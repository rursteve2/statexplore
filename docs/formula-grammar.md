# Discovery candidate grammar

Defined before any search runs (§10.3). Implemented in `discovery/space.py`; verified in `tests/test_discovery.py`.

**Inputs:** outcome *Y*; candidate predictors *P* (ordered as selected, ≤ 39); required covariates *R*;
maximum distinct predictors *k* (2–6; required covariates count toward it); maximum interaction order (2 or 3).

**Candidate** = (focal interaction *F*, covariate set *C*), with |F| ∈ {2, 3}, F ⊆ P, C ⊆ P \ F, |F| + |C| + |R| ≤ k.

- **Full model:** every non-empty subset of *F* (the complete hierarchy) + each variable in *C* + each in *R*, as main effects.
  For example, F = {X1, X2, X3} gives X1 + X2 + X3 + X1:X2 + X1:X3 + X2:X3 + X1:X2:X3.
- **Reduced model:** the full model without the focal term (lower-order terms are kept).
- **Focal test:** nested comparison of reduced vs full: F-test (linear) or likelihood-ratio χ² (logistic).
  Multi-level factors give multi-df tests. Single-df focal terms also report the coefficient and its CI.

**Covariate modes**
- *Focal* (default): C = ∅. Count = C(p,2) [+ C(p,3)].
- *Subsets*: every S ⊆ P with 2 ≤ |S| ≤ k − |R|, every focal F ⊆ S, C = S \ F.
  Count = Σ_s C(p,s)·[C(s,2) + C(s,3)]. The same focal interaction under different covariate sets is a
  **distinct candidate analysis** and enters the multiple-testing family separately.

**Order:** focal mode lists pairs, then triples, lexicographically by candidate position. Subset mode goes
by subset size, then subset (lexicographic), then focal pairs, then triples. The index is stable, so runs can be resumed.

**Uniqueness:** candidates are unique by construction; `candidate_key` = focal + sorted term set (tested).

**Not included:** models with several simultaneous focal interactions, interactions involving
required covariates, mediation, random effects (Phase 5).

**Screening:** a candidate is *invalid* (recorded, not tested) when it exceeds the parameter limit,
leaves no residual df, has an empty focal factor cell, or has a rank-deficient design.
*Flags* (tested, but excluded from support tier A/B): sparse focal cells (< 5), |r| > .9 between
continuous predictors, separation, non-convergence, < 10 events per parameter, R warnings.
