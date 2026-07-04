"""Pareto dominance, non-dominated sorting, crowding, and hypervolume.

The mathematical core of multi-objective optimization, used by both the NSGA-II
engine and the DBTL loop's progress reporting. All objectives are framed as
**maximize** (the oracle returns "higher is better" scores). Pure numpy — no
torch — so it runs and tests anywhere.

Hypervolume (the standard multi-objective progress metric) is computed exactly
for 2 and 3 objectives, which covers our [stability, solubility, binding] case.
"""
from __future__ import annotations


def dominates(a, b) -> bool:
    """True if ``a`` Pareto-dominates ``b`` (maximization).

    ``a`` dominates ``b`` if it is at least as good in every objective and
    strictly better in at least one.
    """
    at_least_as_good = all(ai >= bi for ai, bi in zip(a, b))
    strictly_better = any(ai > bi for ai, bi in zip(a, b))
    return at_least_as_good and strictly_better


def non_dominated_sort(scores) -> list[list[int]]:
    """Fast non-dominated sort (Deb et al., NSGA-II).

    Args:
        scores: list of objective vectors (maximization).

    Returns:
        A list of fronts; each front is a list of indices into ``scores``.
        Front 0 is the Pareto-optimal set.
    """
    n = len(scores)
    dominated_by: list[list[int]] = [[] for _ in range(n)]  # solutions p dominates
    dom_count = [0] * n  # how many dominate p
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(scores[p], scores[q]):
                dominated_by[p].append(q)
            elif dominates(scores[q], scores[p]):
                dom_count[p] += 1
        if dom_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        nxt: list[int] = []
        for p in fronts[i]:
            for q in dominated_by[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()  # last is empty
    return fronts


def pareto_front_indices(scores) -> list[int]:
    """Indices of the non-dominated (Pareto-optimal) solutions (maximization)."""
    if not scores:
        return []
    return non_dominated_sort(scores)[0]


def crowding_distance(scores, indices=None) -> dict:
    """NSGA-II crowding distance for a set of solutions.

    Returns a dict ``index -> distance`` (boundary points get +inf), used to
    preserve diversity when selecting within a front.
    """
    if indices is None:
        indices = list(range(len(scores)))
    if not indices:
        return {}
    m = len(scores[indices[0]])
    distance = {i: 0.0 for i in indices}
    for obj in range(m):
        ordered = sorted(indices, key=lambda i: scores[i][obj])
        lo = scores[ordered[0]][obj]
        hi = scores[ordered[-1]][obj]
        distance[ordered[0]] = float("inf")
        distance[ordered[-1]] = float("inf")
        span = hi - lo
        if span <= 0:
            continue
        for k in range(1, len(ordered) - 1):
            prev_s = scores[ordered[k - 1]][obj]
            next_s = scores[ordered[k + 1]][obj]
            distance[ordered[k]] += (next_s - prev_s) / span
    return distance


# --- hypervolume -------------------------------------------------------------


def _hv2d_min(points, ref) -> float:
    """2D hypervolume for minimization; ``ref`` is the upper-bound (worst) point.

    Filters to points that beat the reference and reduces to the 2D
    non-dominated staircase first, so dominated projections cannot subtract
    area. (This matters when the function is called on the projected active set
    of a 3D sweep, which is not guaranteed non-dominated.)
    """
    rx, ry = ref[0], ref[1]
    pts = sorted((p for p in points if p[0] < rx and p[1] < ry), key=lambda p: (p[0], p[1]))
    hv = 0.0
    last_y = ry
    for x, y in pts:
        if y >= last_y:
            continue  # dominated in 2D by an already-counted point
        hv += (rx - x) * (last_y - y)
        last_y = y
    return hv


def _hv3d_min(points, ref) -> float:
    """3D hypervolume for minimization via a z-axis sweep of 2D slices."""
    pts = sorted(points, key=lambda p: p[2])  # z ascending
    hv = 0.0
    for i, p in enumerate(pts):
        z_i = p[2]
        z_next = pts[i + 1][2] if i + 1 < len(pts) else ref[2]
        if z_i >= ref[2]:
            break
        active = [(q[0], q[1]) for q in pts[: i + 1]]
        area = _hv2d_min(active, (ref[0], ref[1]))
        hv += area * (z_next - z_i)
    return hv


def hypervolume(scores, ref, maximize: bool = True) -> float:
    """Hypervolume dominated by ``scores`` relative to reference point ``ref``.

    Supports 2 or 3 objectives (our case is 3). For maximization the reference
    should be a lower bound ("worst" point); objectives are negated internally to
    a minimization problem.

    Returns 0.0 if no point dominates the reference.
    """
    if not scores:
        return 0.0
    m = len(ref)
    if m not in (2, 3):
        raise ValueError("hypervolume supports 2 or 3 objectives.")
    sign = -1.0 if maximize else 1.0
    pts = [tuple(sign * v for v in s) for s in scores]
    r = tuple(sign * v for v in ref)
    # Keep only points that dominate the reference (strictly better in all dims).
    pts = [p for p in pts if all(pi < ri for pi, ri in zip(p, r))]
    if not pts:
        return 0.0
    # Reduce to the non-dominated set (minimization) for efficiency/correctness.
    nd_idx = pareto_front_indices([tuple(-pi for pi in p) for p in pts])
    pts = [pts[i] for i in nd_idx]
    return _hv2d_min(pts, r) if m == 2 else _hv3d_min(pts, r)
