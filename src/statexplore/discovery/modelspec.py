"""Canonical model specifications (X §2, §14; M §7.1).

A model's fixed part is a set of *terms*; a term is a frozenset of variable names ({x} is a main
effect, {x, m} a two-way interaction). Specs are hierarchically closed: every lower-order term of an
interaction is present. Two formulas are the same model exactly when their closed term sets are
equal, so `y ~ X1*X2 + X1 + X3` and `y ~ X3 + X2:X1 + X1 + X2` share one canonical key.

Complexity is counted three ways and never from formula text:
  distinct predictors  — unique variables (a categorical variable counts once),
  model terms          — main effects + interaction terms,
  fixed coefficients   — columns of the design matrix (intercept included; a k-level factor gives
                         k - 1 columns per term it enters). R reports the exact count per fit.
Grouping variables (participant ID) are never fixed-effect predictors; random-effect parameters are
counted separately (`random_parameters`).
"""
from __future__ import annotations

import re
from itertools import combinations
from typing import Callable, Iterable, Iterator

Term = frozenset


def close(terms: Iterable[Iterable[str]]) -> frozenset[frozenset[str]]:
    """Hierarchical closure: every non-empty subset of every term."""
    out: set[frozenset[str]] = set()
    for t in terms:
        t = tuple(dict.fromkeys(t))
        for k in range(1, len(t) + 1):
            out.update(frozenset(c) for c in combinations(t, k))
    return frozenset(out)


def is_hierarchical(terms: Iterable[Iterable[str]]) -> bool:
    ts = {frozenset(t) for t in terms}
    return ts == set(close(ts))


def sort_terms(terms: Iterable[Iterable[str]], order: list[str] | None = None) -> list[tuple[str, ...]]:
    """Main effects first, then two-way, then three-way; within an order by variable position."""
    pos = {v: i for i, v in enumerate(order or [])}
    key_v = lambda v: (pos.get(v, len(pos)), v)  # noqa: E731
    tup = [tuple(sorted(t, key=key_v)) for t in terms]
    return sorted(tup, key=lambda t: (len(t), [key_v(v) for v in t]))


def canonical_key(terms: Iterable[Iterable[str]]) -> str:
    return " + ".join(":".join(t) for t in sort_terms(close(terms)))


def predictors(terms: Iterable[Iterable[str]]) -> list[str]:
    return sorted({v for t in terms for v in t})


def maximal(terms: Iterable[Iterable[str]]) -> list[frozenset[str]]:
    """Terms not contained in another term of the model."""
    ts = {frozenset(t) for t in terms}
    return [t for t in ts if not any(t < u for u in ts)]


def counts(terms: Iterable[Iterable[str]], levels: dict[str, int] | None = None) -> dict[str, int]:
    """Distinct predictors, term counts by order, total terms; with `levels` (number of categories
    per categorical variable; continuous variables omitted or 1) also the fixed coefficients of a
    full-rank treatment-coded design including the intercept."""
    ts = close(terms)
    out = {"n_predictors": len(predictors(ts)),
           "n_main": sum(len(t) == 1 for t in ts),
           "n_two_way": sum(len(t) == 2 for t in ts),
           "n_three_way": sum(len(t) == 3 for t in ts),
           "n_higher": sum(len(t) > 3 for t in ts),
           "n_terms": len(ts)}
    if levels is not None:
        out["n_fixed"] = fixed_coefficients(ts, levels)
    return out


def fixed_coefficients(terms: Iterable[Iterable[str]], levels: dict[str, int]) -> int:
    n = 1
    for t in close(terms):
        cols = 1
        for v in t:
            k = levels.get(v, 1)
            cols *= (k - 1) if k > 1 else 1
        n += cols
    return n


RANDOM_STRUCTURES = {
    # key: (random-effect variances, random-effect covariances, residual/dispersion parameters)
    "ri": (1, 0, 1), "rs": (2, 1, 1), "rs_uncorr": (2, 0, 1),
    "nlme_ri": (1, 0, 1), "nlme_rs": (2, 1, 1), "ar1": (1, 0, 2), "car1": (1, 0, 2),
    "nlme_diag": (2, 0, 1), "diag_ar1": (2, 0, 2), "rs_ar1": (2, 1, 2), "diag_car1": (2, 0, 2), "rs_car1": (2, 1, 2),
    "glmer_ri": (1, 0, 0), "glmer_rs": (2, 1, 0), "clmm_ri": (1, 0, 0), "clmm_rs": (2, 1, 0),
}


def random_parameters(structure: str, n_times: int | None = None, clustered: bool = False) -> dict[str, int]:
    """Random-effect and residual covariance parameters of a longitudinal structure, counted
    separately from fixed effects (X §2C). mmrm structures depend on the number of occasions. A higher-level cluster
    adds one random-intercept variance."""
    if structure == "mmrm_us" and n_times:
        return {"re_variances": 0, "re_covariances": 0, "residual": n_times * (n_times + 1) // 2}
    if structure == "mmrm_cs":
        return {"re_variances": 0, "re_covariances": 0, "residual": 2}
    if structure == "mmrm_ar1":
        return {"re_variances": 0, "re_covariances": 0, "residual": 2}
    var, cov, res = RANDOM_STRUCTURES.get(structure, (0, 0, 0))
    return {"re_variances": var + (1 if clustered else 0), "re_covariances": cov, "residual": res}


def display(terms: Iterable[Iterable[str]], outcome: str = "y", name: Callable[[str], str] = str,
            order: list[str] | None = None, extra: Iterable[str] = ()) -> str:
    """Compact R-style formula. Interaction blocks whose variables are disjoint are written
    `a * b`; overlapping interactions are written out (`a + b + c + a:b + b:c`)."""
    ts = close(terms)
    inter = [t for t in maximal(ts) if len(t) > 1]
    disjoint = all(not (a & b) for a, b in combinations(inter, 2))
    pos = {v: i for i, v in enumerate(order or predictors(ts))}
    key = lambda t: (min(pos.get(v, 1e9) for v in t), len(t))  # noqa: E731
    vkey = lambda v: pos.get(v, 1e9)  # noqa: E731
    if disjoint:
        covered = {v for t in inter for v in t}
        blocks = sorted(inter, key=key)
        parts = [" * ".join(name(v) for v in sorted(t, key=vkey)) for t in blocks]
        parts += [name(v) for v in sorted((next(iter(t)) for t in ts if len(t) == 1 and not (t & covered)), key=vkey)]
    else:
        parts = [":".join(name(v) for v in t) for t in sort_terms(ts, sorted(pos, key=vkey))]
    parts += list(extra)
    return f"{outcome} ~ {' + '.join(parts) if parts else '1'}"


# ---- parsing (for tests and user-entered formulas) ----------------------------------------

_TOKEN = re.compile(r"\s*([A-Za-z_.][A-Za-z0-9_.]*|\d+|[()+*:^])")


def parse(rhs: str) -> frozenset[frozenset[str]]:
    """Parse a formula right-hand side with +, *, :, parentheses and (a + b + c)^k into the
    closed canonical term set. `y ~` prefixes are accepted. Raises ValueError on bad input."""
    if "~" in rhs:
        rhs = rhs.split("~", 1)[1]
    tokens = []
    pos = 0
    rhs = rhs.strip()
    while pos < len(rhs):
        m = _TOKEN.match(rhs, pos)
        if not m:
            raise ValueError(f"Cannot parse formula near {rhs[pos:]!r}")
        tokens.append(m.group(1))
        pos = m.end()
    it = _Parser(tokens)
    terms = it.sum()
    if it.peek() is not None:
        raise ValueError(f"Unexpected {it.peek()!r} in formula")
    return close(t for t in terms if t)


class _Parser:
    """sum := prod ('+' prod)* ; prod := inter ('*' inter)* ; inter := power (':' power)* ;
    power := atom ('^' int)? ; atom := name | '1' | '(' sum ')'. Values are sets of terms."""

    def __init__(self, tokens: list[str]):
        self.t, self.i = tokens, 0

    def peek(self) -> str | None:
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self) -> str:
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of formula")
        self.i += 1
        return tok

    def sum(self) -> set[frozenset[str]]:
        out = self.prod()
        while self.peek() == "+":
            self.take()
            out |= self.prod()
        return out

    def prod(self) -> set[frozenset[str]]:
        out = self.inter()
        while self.peek() == "*":
            self.take()
            rhs = self.inter()
            out = out | rhs | {a | b for a in out for b in rhs}
        return out

    def inter(self) -> set[frozenset[str]]:
        out = self.power()
        while self.peek() == ":":
            self.take()
            rhs = self.power()
            out = {a | b for a in out for b in rhs}
        return out

    def power(self) -> set[frozenset[str]]:
        base = self.atom()
        if self.peek() == "^":
            self.take()
            k = int(self.take())
            out = set(base)
            for _ in range(k - 1):
                out = out | {a | b for a in out for b in base}
            return {t for t in out if len(t) <= k}
        return base

    def atom(self) -> set[frozenset[str]]:
        tok = self.take()
        if tok == "(":
            out = self.sum()
            if self.take() != ")":
                raise ValueError("Missing ')'")
            return out
        if tok in ("1", "0"):
            return {frozenset()}
        if not re.match(r"[A-Za-z_.]", tok):
            raise ValueError(f"Unexpected {tok!r} in formula")
        return {frozenset([tok])}


# ---- enumeration of hierarchical interaction sets ------------------------------------------

def interaction_sets(variables: tuple[str, ...], max_order: int = 2) -> Iterator[tuple[tuple[str, ...], ...]]:
    """Every hierarchical set of interaction terms (orders 2..max_order) on `variables`, all main
    effects implied. Deterministic order: by the set of two-way terms (fewest first, then
    lexicographic by position), then by the three-way terms allowed by that set (likewise).
    The empty set (main effects only) comes first."""
    pairs = list(combinations(variables, 2))
    for k in range(len(pairs) + 1):
        for chosen in combinations(pairs, k):
            if max_order < 3:
                yield chosen
                continue
            have = {frozenset(p) for p in chosen}
            triples = [t for t in combinations(variables, 3)
                       if all(frozenset(p) in have for p in combinations(t, 2))]
            for j in range(len(triples) + 1):
                for tri in combinations(triples, j):
                    yield chosen + tri


_N_SETS: dict[tuple[int, int], int] = {}


def n_interaction_sets(size: int, max_order: int = 2) -> int:
    """Exact number of hierarchical interaction sets on `size` variables (incl. the empty set).
    Two-way only: 2^C(size, 2). With three-way terms: sum over two-way sets of 2^(triangles)."""
    key = (size, max_order)
    if key not in _N_SETS:
        if max_order < 3 or size < 3:
            _N_SETS[key] = 2 ** (size * (size - 1) // 2)
        else:
            vs = tuple(range(size))
            pairs = list(combinations(vs, 2))
            triples = list(combinations(vs, 3))
            tri_pairs = [[pairs.index(p) for p in combinations(t, 2)] for t in triples]
            total = 0
            for mask in range(2 ** len(pairs)):
                n_tri = sum(all(mask >> i & 1 for i in tp) for tp in tri_pairs)
                total += 2 ** n_tri
            _N_SETS[key] = total
    return _N_SETS[key]
