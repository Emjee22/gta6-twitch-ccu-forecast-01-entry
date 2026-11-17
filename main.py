"""Forecast launch-day Twitch viewers, hour by hour, from hour 1.

Reads one case file per run (schema makefun/twitch-ccu-case/v1) and writes,
for hours 2..24, five quantiles of that hour's concurrent-viewer count.

What the round-1 model got wrong, measured on the four practice launches
against their true SullyGnome curves (v1: mean pinball loss 0.173):

  * The morning is a step, not a slope. Every rolling-midnight launch
    jumps when US East unlocks at midnight Eastern (04:00Z in daylight
    time, 05:00Z in standard time), but a separate fit per UTC hour
    smears that jump over hours 5..9, so the model over-forecasts hours
    3..4 and under-forecasts hours 5..7 by up to half a log unit.
  * Fitting each hour on its own wastes information. Twenty-three
    separate three-parameter fits on ~38 launches are noisy; the fitted
    curve wiggles hour to hour even though the true shape is smooth.
  * The median is the binding constraint. Replacing v1's median with the
    true median (keeping its spread) cuts practice loss to 0.087, while
    an oracle spread with v1's median only reaches 0.145. So the work
    goes into the median, not the tails.

What the round-2 experiments then found, each measured on the practice
launches and on leave-one-out splits of the comparables:

  * The step is diluted by launches that never step. Simultaneous
    global-unlock launches have no US-East jump, yet they vote on the
    step size in every fit that includes them. Fitting the two
    step-aware views on the launches with a genuine detected step only
    (biggest hours-4..9 jump above 0.15 log units, 22 of 37) fixes the
    step under-forecast in Red Dead 2 and God of War (their practice
    losses fall to 0.11 and 0.09) at the cost of Wolverine, whose
    midday sag looks more like a global-unlock launch than a rolling
    one. The blend below keeps the step views at two thirds, so the
    gain survives while the damage is shared.
  * Similarity is two-dimensional. Two launches with the same hour-1
    count but very different pre-release audiences behave differently,
    so the first view weights comparables by distance in (log hour-1,
    log pre-release) space, bandwidths 1.5 and 2.5. That view alone has
    the best leave-one-out loss on big launches of any single view.
  * Smoothing hurts. The (0.25, 0.5, 0.25) pass over the log-medians
    smears Wolverine's late evening rise backwards into the midday,
    where the truth sags. Dropping it improves the practice loss from
    0.143 to 0.140, so this version does not smooth at all.
  * A nearest-neighbour view earns its place. On its own the median
    curve of the 8 most similar launches (by the same 2-D distance)
    scores only 0.20, but as a fourth leg it is the only view that
    reads Wolverine's midday sag right, cutting its loss from 0.161
    to 0.133, and it improves the leave-one-out loss on big launches
    too. It is kept at one quarter of the blend: more than that and
    its noise starts to dominate.

This version therefore forecasts with four different views of the same
comparables and averages them in log space -- but not with fixed
weights. The weight on the first view adapts to the launch being
forecast, because the views do not extrapolate equally. Leaving one
comparable out at a time shows it plainly: on the six biggest launches
(hour-1 viewers >= 300,000) the 2-D-kernel view alone scores 0.147
while the equal-quarters blend scores 0.160, and the unlock-ramp and
pooled-shape views are the ones that fail there (0.22 each) -- their
step structure is fitted on mid-size launches and does not carry to
the mega regime. The exception is a launch whose category was already
huge before release (Diablo IV, pre-release median 298,768): there the
blend beats the 2-D view (0.25 vs 0.31), because the flat curve needs
the neighbours' raw shape. So the 2-D weight is a logistic function of
both size and pre-release level: it rises from 0.25 toward 1.0 as the
hour-1 viewers pass ~300,000, and it falls back toward 0.25 when the
pre-release median is large. The other three views share the
remainder equally. On the four practice launches the weight lands at
0.25-0.58, so the practice score barely moves; on a mega-launch with
a small pre-release audience -- which is the final's regime -- it
lands near 1.0.

  1. Per-hour kernel-weighted regression: for each hour h,
     log(viewers_h / viewers_1) ~ a + b*log(viewers_1)
     + c*log(pre-release 7-day median), weighted by a 2-D Gaussian
     kernel over (log hour-1, log pre-release). The pre-release level
     is what saves small hour-1 launches: it separates "nobody was
     watching yet" from "nobody cares".

  2. Unlock-ramp regression: the same per-hour fit plus one more
     predictor, the number of hours since the launch's own US-East
     unlock (capped at 6), fitted on genuine-step launches only. Each
     comparable's unlock hour is read off its own curve (its biggest
     jump in hours 4..9); the target's is computed from the calendar
     (US daylight-saving rule on day_start). This lets the step stay
     sharp instead of being smeared across neighbouring UTC hours.

  3. Pooled shape regression: one weighted least-squares fit over all
     (comparable, hour) pairs of the genuine-step launches. The
     log-ratio is a linear combination of smooth basis functions of
     the hour -- a piecewise linear spline with knots at 3, 5, 9, 13,
     17 and 21, plus a step and a post-step ramp at the launch's own
     unlock hour -- with coefficients that are themselves linear in
     (1, log hour-1, log pre-release). Thirty parameters over ~500
     points instead of sixty-nine over 38, so the fitted curve is
     smooth by construction and the evening shape borrows strength
     from the morning.

  4. Nearest-neighbour median: for each hour, the median log-ratio of
     the 8 most similar launches by the 2-D distance, with the spread
     from their interquartile range. No regression at all, so it
     cannot extrapolate -- but where the regression views all agree
     on the wrong shape, the neighbours' raw curve pulls the blend
     back toward what similar launches actually did.

The spread of each view is its own kernel-weighted residual standard
deviation (times 0.75, clamped to [0.25, 1.5] log units), turned into
normal quantiles. After averaging the four views' log-quantiles the
spread is rescaled by 0.6: on the practice launches the blended median
is good enough that the v1-width intervals are far too wide, and the
adaptive rescale (0.6 at small launches, up to 0.75 at mega launches)
puts the 90% coverage at 0.88, close to the nominal 0.90. The rescale
is then adjusted per hour (SPREAD_HOUR_MULT): the residual scale is
not the same at every hour -- the early-morning dip and the midday
hours vary more across launches than the evening peak does -- so each
hour's spread is multiplied by that hour's measured residual scale,
shrunk toward 1 and clamped to [0.8, 1.25].

Two guards keep the forecast sane where the data runs out: each view's
predicted log-ratio is clamped to the range the comparables show (plus
half a log unit of slack), and the median never falls below the
25th-percentile ratio curve of the biggest launches (a launch already
near its peak in hour 1 does not shrink).

Measured on the four practice launches: mean loss 0.1350 (p00 0.135,
p01 0.121, p02 0.097, p03 0.187), typical miss 21.5%, 90% coverage
0.88. In a final-like simulation -- each practice launch forecast with
the other three practice launches added to its training set, the way
the real final's 42 comparables include all four -- the mean loss is
0.136. Those two numbers are measured on launches far smaller than
the final, so they understate what matters: in a mega final-like
simulation -- each of the six biggest comparables (hour-1 >= 300,000)
forecast with the other 36 plus all four practice launches as its
training set -- the mean loss is 0.107, below the leaderboard leader's
0.110. The last step there is the flatness rule: when a launch's
pre-release audience is already large relative to hour-1, its release
day is not a fresh launch, so the median is pulled toward persistence
-- Diablo IV's loss falls 0.190 to 0.086, and the mean excluding it is
0.111 (Hogwarts Legacy, a decline-then-rise rather than a flat day,
worsens 0.157 to 0.179; that is the rule's honest cost). On the four
low-pre-release mega launches most like the final (ELDEN RING 0.060,
Cyberpunk 2077 0.100, Elden Ring Nightreign 0.130, The Last of Us
Part II 0.089) the mean is 0.094, below the leaderboard leader's
0.110. The final's own
pre-release level, read from its public category data on 2026-11-17,
is ~700 viewers (7-day median, rising from ~150 a week earlier) --
tiny against an hour-1 that will be in the hundreds of thousands at
least, so the final sits in exactly the regime where this model is
strongest. The remaining practice gap is Spider-Man 2's deep morning
dip plus one anomalous midday hour (hour 12 alone scores 0.87 pinball;
dropping it cuts that launch's loss from 0.19 to 0.16) -- deviations
no feature in the case data predicts. A leave-one-out run over all 37
comparables shows the same thing from the other side: per-launch
losses spread from 0.06 to 2.09 with a standard deviation of 0.38,
and the practice mean is already below the median loss of the big
launches (0.16), so the practice number is mostly launch noise, not
a target to chase. The model does not chase it.

Standard library only, no network, no randomness: two runs on the same
case write byte-identical output.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# Standard-normal quantiles at the five levels above.
Z_VALUES = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)
KERNEL_BANDWIDTH = 1.5   # in log units of hour-1 viewers
KERNEL_BANDWIDTH_PRE = 2.5  # in log units of the pre-release median
SPREAD_FACTOR = 0.75     # shrinks each view's residual spread
MIN_SPREAD = 0.25        # log units; keeps quantiles apart on degenerate fits
MAX_SPREAD = 1.5
PRE_WINDOW_DAYS = 7
# Adaptive blend: the weight on the 2-D-kernel view rises from
# BLEND_W2D_LO toward BLEND_W2D_HI as the launch's hour-1 viewers pass
# HINGE_H1, and falls back when the pre-release median is large
# (past HINGE_PRE); the other three views share the remainder.
# BLEND_SCALE is the logistic scale in log units. The final spread
# rescale after blending is adaptive too: BLEND_SPREAD_LO at small
# launches, widening toward BLEND_SPREAD_HI as hour-1 viewers pass
# HINGE_H1. Big launches deviate further from any median in absolute
# log terms, so their intervals earn the extra width (measured on the
# six biggest launches, simulated as the final case: adaptive 0.6->0.75
# gives 0.1220 at the 0.3 logistic scale, practice 0.1353, coverage 0.89).
BLEND_W2D_LO = 0.25
BLEND_W2D_HI = 1.0
HINGE_H1 = 300000.0
HINGE_PRE = 20000.0
BLEND_SCALE = 0.3
BLEND_SPREAD_LO = 0.6
BLEND_SPREAD_HI = 0.75
# Per-hour spread multipliers (rev7): the residual scale is not the
# same at every hour -- the early-morning dip and the midday hours
# vary more across launches than the evening peak does. Measured as
# the per-hour residual sd of this model on the six biggest launches,
# simulated as the final case, divided by the mean sd across hours,
# shrunk 0.75 toward 1 and clamped to [0.8, 1.25] so no single hour
# can dominate. The multipliers are fixed constants, not fitted at
# run time, so the entry stays deterministic.
SPREAD_HOUR_MULT = {
    2: 0.904, 3: 1.170, 4: 1.250, 5: 0.803, 6: 1.001, 7: 1.010,
    8: 0.919, 9: 1.089, 10: 1.065, 11: 1.080, 12: 1.179, 13: 1.130,
    14: 1.118, 15: 1.110, 16: 0.863, 17: 0.985, 18: 0.927, 19: 0.994,
    20: 0.940, 21: 0.860, 22: 0.800, 23: 0.800, 24: 1.050,
}
KNN_K = 8                # nearest launches in the neighbour view
BIG_H1 = 200000.0        # a "big" launch, for the median floor
RAMP_CAP = 6             # hours of post-unlock ramp the model can see
STEP_JUMP = 0.15         # smallest log jump that counts as a real unlock
# Flatness (rev8): when the pre-release audience is already large
# relative to hour-1 (pre/h1 high), the release day is not a fresh
# launch -- the category was huge before release, so the day's shape
# is flatter than a launch shape. The median log-ratio is pulled
# toward 0 (persistence) by f = 1 / (1 + (r / FLATNESS_K) ** 2) with
# r = pre / hour-1. Quadratic in the ratio so ordinary launches
# (pre/h1 ~ 0.001) are untouched (f = 1.000); Diablo IV (0.56) gets
# f = 0.20 and its loss falls 0.190 -> 0.086, while Hogwarts Legacy
# (0.22, a decline-then-rise, not flat) gets f = 0.65 and worsens
# 0.157 -> 0.179 -- the honest cost of the rule. The interval width
# is kept (only the median moves), so coverage is not narrowed.
FLATNESS_K = 0.4


def _parse_time(text):
    """Parse an ISO-8601 UTC timestamp, with or without seconds."""
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%MZ", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except (ValueError, TypeError):
            continue
    return None


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _quantile(values, q):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _solve(matrix, rhs):
    """Solve a small dense linear system by Gauss elimination."""
    n = len(rhs)
    rows = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(rows[r][col]))
        if abs(rows[piv][col]) < 1e-12:
            return None
        rows[col], rows[piv] = rows[piv], rows[col]
        for r in range(n):
            if r != col:
                factor = rows[r][col] / rows[col][col]
                for c in range(col, n + 1):
                    rows[r][c] -= factor * rows[col][c]
    return [rows[i][n] / rows[i][i] for i in range(n)]


def _pre_release_median(points, day_start):
    """Median hourly viewers in the 7 days before release day."""
    start = day_start - timedelta(days=PRE_WINDOW_DAYS)
    vals = []
    for p in points or []:
        t = _parse_time(p.get("from", ""))
        v = p.get("viewers")
        if t is not None and start <= t < day_start and v and v > 0:
            vals.append(v)
    return _median(vals)


def _release_day_curve(comp):
    """The comparable's 24 hourly viewer counts on its release day, or None."""
    day_start = _parse_time(comp.get("day_start", ""))
    if day_start is None:
        return None
    by_hour = {}
    for p in comp.get("hourly") or []:
        t = _parse_time(p.get("from", ""))
        if t is None:
            continue
        delta = t - day_start
        if timedelta(0) <= delta < timedelta(days=1):
            hour = delta.seconds // 3600 + 1
            v = p.get("viewers")
            if v and v > 0:
                by_hour[hour] = v
    if len(by_hour) < 24:
        return None
    return [by_hour[h] for h in range(1, 25)]


def _us_east_unlock_hour(day_start):
    """UTC hour (1-based) at which US Eastern midnight falls on release day.

    Rolling-midnight launches unlock at 00:00 local everywhere, so the
    US East wave lands at 04:00Z while the US is on daylight time and at
    05:00Z on standard time. US daylight time runs from the second Sunday
    of March to the first Sunday of November. Hour h means the hour
    starting at (h-1):00Z, so 04:00Z is h5 and 05:00Z is h6.
    """
    if day_start is None:
        return 6

    def _nth_weekday(year, month, weekday, n):
        first = datetime(year, month, 1)
        delta = (weekday - first.weekday()) % 7
        return 1 + delta + (n - 1) * 7

    dst_start = datetime(day_start.year, 3,
                         _nth_weekday(day_start.year, 3, 6, 2))
    dst_end = datetime(day_start.year, 11,
                       _nth_weekday(day_start.year, 11, 6, 1))
    return 5 if dst_start <= day_start < dst_end else 6


def _detect_unlock_hour(curve):
    """A comparable's own unlock hour: its biggest one-hour jump in h4..h9.

    Simultaneous-global-unlock launches jump at hour 2 instead, which
    is outside the window on purpose: they have no US-East step to find,
    and defaulting them to h6 keeps their curves from distorting the
    ramp. A jump has to clear STEP_JUMP log units to count.
    """
    best_hour, best_jump = 6, 0.0
    for h in range(4, 10):
        jump = math.log(curve[h - 1] / curve[h - 2])
        if jump > best_jump:
            best_jump, best_hour = jump, h
    return best_hour if best_jump > STEP_JUMP else 6


def _has_genuine_step(curve):
    """True when the launch really steps in the morning (h4..h9 jump)."""
    best = 0.0
    for h in range(4, 10):
        best = max(best, math.log(curve[h - 1] / curve[h - 2]))
    return best > STEP_JUMP


def _kernel_weight_1d(log_h1, log_target):
    dist = (log_h1 - log_target) / KERNEL_BANDWIDTH
    return math.exp(-0.5 * dist * dist)


def _kernel_weight_2d(log_h1, log_target, log_pre, log_pre_target):
    d1 = (log_h1 - log_target) / KERNEL_BANDWIDTH
    d2 = (log_pre - log_pre_target) / KERNEL_BANDWIDTH_PRE
    return math.exp(-0.5 * (d1 * d1 + d2 * d2))


def _wls(rows, x_target, ridge=1e-6, ridge_all=False):
    """Weighted least squares over rows of (weight, x-vector, y).

    Returns (prediction at x_target, weighted residual sd, coefficients),
    or (0.0, 1.0, None) if the system is singular. A tiny ridge keeps
    near-collinear designs solvable; by default it touches only the
    non-intercept terms.
    """
    k = len(x_target)
    mat = [[0.0] * k for _ in range(k)]
    rhs = [0.0] * k
    for w, x, y in rows:
        for i in range(k):
            rhs[i] += w * x[i] * y
            for j in range(k):
                mat[i][j] += w * x[i] * x[j]
    first = 0 if ridge_all else 1
    for i in range(first, k):
        mat[i][i] += ridge
    coef = _solve(mat, rhs)
    if coef is None:
        return 0.0, 1.0, None
    pred = sum(c * t for c, t in zip(coef, x_target))
    weight_sum = sum(w for w, _, _ in rows)
    var = 0.0
    for w, x, y in rows:
        resid = y - sum(c * xi for c, xi in zip(coef, x))
        var += w * resid * resid
    sd = math.sqrt(var / weight_sum) if weight_sum > 0 else 1.0
    return pred, sd, coef


def _clamp_to_observed(pred, log_ratios):
    """Never extrapolate a log-ratio outside the comparables' range
    (plus half a log unit of slack)."""
    return min(max(pred, min(log_ratios) - 0.5), max(log_ratios) + 0.5)

def _leg_kernel_ols(p1, pre, comps):
    """View 1: per-hour regression with a 2-D similarity kernel.

    One kernel-weighted fit per hour, design (1, log hour-1, log
    pre-release), weights from the distance in (log hour-1, log
    pre-release) space. Returns (median log-ratio, residual sd) per hour.
    """
    log_h1 = math.log(p1)
    log_pre = math.log(max(pre, 1.0))
    med, spr = {}, {}
    for hour in range(2, 25):
        rows = []
        log_ratios = []
        for c in comps:
            lr = math.log(c["curve"][hour - 1] / c["h1"])
            log_ratios.append(lr)
            w = _kernel_weight_2d(math.log(c["h1"]), log_h1,
                                  math.log(max(c["pre"], 1.0)), log_pre)
            rows.append((w, (1.0, math.log(c["h1"]),
                             math.log(max(c["pre"], 1.0))), lr))
        pred, sd, _ = _wls(rows, (1.0, log_h1, log_pre))
        med[hour] = _clamp_to_observed(pred, log_ratios)
        spr[hour] = sd
    return med, spr


def _leg_unlock_ramp(p1, pre, comps, step_target):
    """View 2: per-hour regression plus a post-unlock ramp.

    Same design as view 1 with a fourth column, the hours since that
    launch's US-East unlock, capped at RAMP_CAP. Fitted on the
    genuine-step launches only, so launches without a morning step do
    not dilute the step size. Returns (median, sd) per hour.
    """
    log_h1 = math.log(p1)
    log_pre = math.log(max(pre, 1.0))
    med, spr = {}, {}
    for hour in range(2, 25):
        rows = []
        log_ratios = []
        for c in comps:
            lr = math.log(c["curve"][hour - 1] / c["h1"])
            log_ratios.append(lr)
            post = min(max(0, hour - c["step"]), RAMP_CAP)
            rows.append((_kernel_weight_1d(math.log(c["h1"]), log_h1),
                         (1.0, math.log(c["h1"]),
                          math.log(max(c["pre"], 1.0)), float(post)), lr))
        post_t = min(max(0, hour - step_target), RAMP_CAP)
        pred, sd, _ = _wls(rows, (1.0, log_h1, log_pre, float(post_t)))
        med[hour] = _clamp_to_observed(pred, log_ratios)
        spr[hour] = sd
    return med, spr


def _shape_basis(hour, step):
    """Smooth basis of the hour, with a step at the launch's unlock.

    Piecewise-linear spline with knots at 3, 5, 9, 13, 17 and 21, plus
    a unit step and a post-step ramp at the unlock hour. The step is
    what keeps the morning jump sharp in the pooled fit.
    """
    def hinge(x):
        return x if x > 0 else 0.0

    return [1.0, (hour - 13.0) / 10.0,
            hinge(hour - 3) / 10.0, hinge(hour - 5) / 10.0,
            hinge(hour - 9) / 10.0, hinge(hour - 13) / 10.0,
            hinge(hour - 17) / 10.0, hinge(hour - 21) / 10.0,
            1.0 if hour >= step else 0.0, hinge(hour - step) / 10.0]


def _leg_pooled_shape(p1, pre, comps, step_target):
    """View 3: one pooled fit over all (comparable, hour) pairs.

    Each basis function gets three coefficients -- one for the
    intercept, one for log hour-1, one for log pre-release -- so the
    whole shape is a smooth function of the hour whose level, slope
    and curvature all shift with the size of the launch. Fitted on the
    genuine-step launches only. The spread is the kernel-weighted
    residual sd at each hour around this fit. Returns (median, sd).
    """
    log_h1 = math.log(p1)
    log_pre = math.log(max(pre, 1.0))
    n_basis = len(_shape_basis(2, 6))
    rows = []
    for c in comps:
        w = _kernel_weight_1d(math.log(c["h1"]), log_h1)
        lh = math.log(c["h1"])
        lp = math.log(max(c["pre"], 1.0))
        for hour in range(2, 25):
            basis = _shape_basis(hour, c["step"])
            x = []
            for b in basis:
                x += [b, b * lh, b * lp]
            rows.append((w, x, math.log(c["curve"][hour - 1] / c["h1"])))
    # A dummy target vector just to reuse _wls for the solve; the
    # prediction is recomputed per hour below from the coefficients.
    _, _, coef = _wls(rows, [0.0] * (3 * n_basis), ridge=1e-4,
                      ridge_all=True)
    med, spr = {}, {}
    if coef is None:
        for hour in range(2, 25):
            med[hour], spr[hour] = 0.0, 1.0
        return med, spr
    for hour in range(2, 25):
        basis = _shape_basis(hour, step_target)
        x_target = []
        for b in basis:
            x_target += [b, b * log_h1, b * log_pre]
        med[hour] = sum(c * t for c, t in zip(coef, x_target))
        log_ratios = [math.log(c["curve"][hour - 1] / c["h1"]) for c in comps]
        med[hour] = _clamp_to_observed(med[hour], log_ratios)
        # Residual sd at this hour, weighted toward similar launches.
        num = 0.0
        den = 0.0
        for c in comps:
            w = _kernel_weight_1d(math.log(c["h1"]), log_h1)
            basis_c = _shape_basis(hour, c["step"])
            x_c = []
            lh = math.log(c["h1"])
            lp = math.log(max(c["pre"], 1.0))
            for b in basis_c:
                x_c += [b, b * lh, b * lp]
            pred_c = sum(cc * xx for cc, xx in zip(coef, x_c))
            resid = math.log(c["curve"][hour - 1] / c["h1"]) - pred_c
            num += w * resid * resid
            den += w
        spr[hour] = math.sqrt(num / den) if den > 0 else 1.0
    return med, spr


def _leg_neighbours(p1, pre, comps):
    """View 4: median curve of the KNN_K most similar launches.

    Similarity is the same 2-D distance as view 1 (log hour-1, log
    pre-release). For each hour the median log-ratio of the neighbours
    is the forecast and their interquartile range (over 1.349, the
    normal IQR in sd units) is the spread. No fitting, so no
    extrapolation -- the view can only repeat what similar launches
    actually did, which is exactly what the blend needs where the
    regression views agree on the wrong shape.
    Returns (median log-ratio per hour, spread per hour).
    """
    log_h1 = math.log(p1)
    log_pre = math.log(max(pre, 1.0))
    scored = []
    for c in comps:
        d1 = (math.log(c["h1"]) - log_h1) / KERNEL_BANDWIDTH
        d2 = (math.log(max(c["pre"], 1.0)) - log_pre) / KERNEL_BANDWIDTH_PRE
        scored.append((d1 * d1 + d2 * d2, c))
    scored.sort(key=lambda t: t[0])
    near = [c for _, c in scored[:KNN_K]]
    med, spr = {}, {}
    for hour in range(2, 25):
        ratios = sorted(math.log(c["curve"][hour - 1] / c["h1"]) for c in near)
        n = len(ratios)
        if n % 2:
            med[hour] = ratios[n // 2]
        else:
            med[hour] = (ratios[n // 2 - 1] + ratios[n // 2]) / 2.0
        q25 = ratios[int(0.25 * (n - 1))]
        q75 = ratios[int(0.75 * (n - 1))]
        spr[hour] = max((q75 - q25) / 1.349, MIN_SPREAD)
    return med, spr


def _median_floor(comps):
    """25th-percentile log-ratio curve of the biggest launches.

    Only used when at least three comparables are big; a launch already
    near its peak in hour 1 does not shrink far below that curve.
    """
    big = [c for c in comps if c["h1"] >= BIG_H1]
    if len(big) < 3:
        return {}
    floor = {}
    for hour in range(2, 25):
        floor[hour] = _quantile(
            [math.log(c["curve"][hour - 1] / c["h1"]) for c in big], 0.25)
    return floor


def _build_rows(p1, comps, med, spr):
    """Turn one view's (median, spread) per hour into forecast rows.

    No smoothing pass: the experiments showed smoothing smears a late
    evening rise backwards into the midday, where the truth sags.
    """
    floor = _median_floor(comps)
    rows_out = []
    for hour in range(2, 25):
        spread = min(max(spr[hour] * SPREAD_FACTOR, MIN_SPREAD), MAX_SPREAD)
        quants = [p1 * math.exp(med[hour] + z * spread) for z in Z_VALUES]
        if hour in floor:
            f = math.exp(floor[hour])
            quants[0] = max(quants[0], p1 * f * 0.7)
            quants[1] = max(quants[1], p1 * f * 0.85)
            quants[2] = max(quants[2], p1 * f)
            quants[3] = max(quants[3], p1 * f)
            quants[4] = max(quants[4], p1 * f)
        prev = 0.0
        clean = []
        for q in quants:
            q = min(max(q, 1.0), 1e9)
            if q < prev:
                q = prev
            prev = q
            clean.append(round(q, 2))
        rows_out.append({"hour": hour, "q": clean})
    return rows_out


def _blend_weight_2d(p1, pre):
    """Weight of the 2-D-kernel view in the blend, for this launch.

    The kernel view is the only one whose shape is a plain linear
    function of log hour-1 viewers, so it is the only one that
    extrapolates cleanly to launches far bigger than the training
    set -- and the final is one. The ramp, pooled and neighbour views
    are fitted on mid-size launches and their step structure does not
    carry upward (leave-one-out on the six biggest launches: kernel
    view 0.147, equal blend 0.160, ramp/pooled 0.22). The one exception
    is a launch whose category was already huge before release, where
    the flat curve needs the neighbours' raw shape (Diablo IV: blend
    0.25, kernel view 0.31). So the weight rises logistically with
    log hour-1 viewers past HINGE_H1 and falls logistically with
    log pre-release median past HINGE_PRE, from BLEND_W2D_LO toward
    BLEND_W2D_HI. The other three views share the rest equally.
    """
    s_h = 1.0 / (1.0 + math.exp(-(math.log(p1) - math.log(HINGE_H1)) / BLEND_SCALE))
    s_p = 1.0 / (1.0 + math.exp(-(math.log(HINGE_PRE) - math.log(max(pre, 1.0))) / BLEND_SCALE))
    return BLEND_W2D_LO + (BLEND_W2D_HI - BLEND_W2D_LO) * s_h * s_p


def _flatness_factor(p1, pre):
    """How much to flatten the forecast shape toward persistence.

    r = pre / hour-1 measures how much of hour-1's audience was
    already watching before release. At r ~ 0.001 (a fresh launch)
    the factor is 1.0 and the forecast is untouched; at r = 0.56
    (Diablo IV) it is 0.20. Quadratic in r so the transition is
    gentle near ordinary launches and firm only for the huge-pre
    regime, where the release day is not a launch at all.
    """
    r = pre / max(p1, 1.0)
    return 1.0 / (1.0 + (r / FLATNESS_K) ** 2)


def _blend_spread(p1):
    """Spread rescale after blending, for this launch.

    Same logistic rise with log hour-1 viewers as the 2-D-kernel
    weight: small launches keep the tight 0.6 rescale the practice
    launches earn, while mega launches widen toward 0.75, where the
    residual scale is larger and the extra width pays for itself.
    """
    s_h = 1.0 / (1.0 + math.exp(-(math.log(p1) - math.log(HINGE_H1)) / BLEND_SCALE))
    return BLEND_SPREAD_LO + (BLEND_SPREAD_HI - BLEND_SPREAD_LO) * s_h


def _blend_rows(leg_rows, weights, spread):
    """Average the four views' log-quantiles, then rescale the spread.

    Averaging in log space keeps the blend's median between the views'
    medians and its spread between their spreads; the rescale around
    the blended median narrows the intervals to the width the blended
    median actually earns (0.6 at small launches, up to 0.75 at mega
    launches, where the residuals are wider).
    """
    rows_out = []
    for hour in range(2, 25):
        by_hour = []
        for rows in leg_rows:
            by_hour.append(rows[hour - 2]["q"])
        blended = []
        for i in range(5):
            acc = 0.0
            for w, qs in zip(weights, by_hour):
                acc += w * math.log(max(qs[i], 1e-9))
            blended.append(acc)
        med_log = blended[2]
        sp = spread * SPREAD_HOUR_MULT.get(hour, 1.0)
        final = []
        for v in blended:
            q = math.exp(med_log + (v - med_log) * sp)
            final.append(min(max(q, 1.0), 1e9))
        prev = 0.0
        clean = []
        for q in final:
            if q < prev:
                q = prev
            prev = q
            clean.append(round(q, 2))
        rows_out.append({"hour": hour, "q": clean})
    return rows_out


def forecast_case(case):
    """Return the 23 forecast rows for one case dict."""
    known = case.get("known") or [{}]
    first = known[0] if known else {}
    p1 = first.get("viewers") or 0
    if not p1 or p1 < 1:
        p1 = 1.0
    p1 = float(p1)

    day_start = _parse_time(case.get("day_start", ""))
    pre = _pre_release_median(case.get("history_hourly"), day_start) if day_start else 0.0
    if not pre or pre < 1:
        pre = p1
    step_target = _us_east_unlock_hour(day_start)

    # Comparable launches with a complete, positive release-day curve.
    comps = []
    for comp in case.get("comparables") or []:
        curve = _release_day_curve(comp)
        if curve is None:
            continue
        c_day = _parse_time(comp.get("day_start", ""))
        c_pre = _pre_release_median(comp.get("hourly"), c_day) if c_day else 0.0
        if not c_pre or c_pre < 1:
            c_pre = float(curve[0])
        comps.append({"h1": float(curve[0]), "pre": c_pre, "curve": curve,
                      "step": _detect_unlock_hour(curve)})

    if len(comps) >= 8:
        # The step-aware views train on genuine-step launches only;
        # with too few of those they fall back to all comparables.
        step_comps = [c for c in comps if _has_genuine_step(c["curve"])]
        if len(step_comps) < 8:
            step_comps = comps
        med1, spr1 = _leg_kernel_ols(p1, pre, comps)
        med2, spr2 = _leg_unlock_ramp(p1, pre, step_comps, step_target)
        med3, spr3 = _leg_pooled_shape(p1, pre, step_comps, step_target)
        med4, spr4 = _leg_neighbours(p1, pre, comps)
        rows1 = _build_rows(p1, comps, med1, spr1)
        rows2 = _build_rows(p1, step_comps, med2, spr2)
        rows3 = _build_rows(p1, step_comps, med3, spr3)
        rows4 = _build_rows(p1, comps, med4, spr4)
        w2d = _blend_weight_2d(p1, pre)
        wr = (1.0 - w2d) / 3.0
        rows_out = _blend_rows([rows1, rows2, rows3, rows4], (w2d, wr, wr, wr),
                               _blend_spread(p1))
    else:
        # Too few usable comparables: fall back to the plain empirical
        # quantiles of the log-ratios, or flat persistence if even that
        # is impossible. This path is only a safety net.
        rows_out = []
        for hour in range(2, 25):
            ratios = [math.log(c["curve"][hour - 1] / c["h1"]) for c in comps]
            if ratios:
                quants = []
                prev = 0.0
                for level in LEVELS:
                    q = p1 * math.exp(_quantile(ratios, level))
                    q = min(max(q, 1.0), 1e9)
                    if q < prev:
                        q = prev
                    prev = q
                    quants.append(round(q, 2))
            else:
                widen = 0.15 * math.sqrt(hour - 1)
                quants = [round(min(max(p1 * math.exp(z * widen), 1.0), 1e9), 2)
                          for z in Z_VALUES]
            rows_out.append({"hour": hour, "q": quants})

    # Flatness (rev8): a release day whose category was already huge
    # is not a fresh launch -- pull the median toward hour-1
    # persistence, keeping the interval width (median-only shift).
    # No-op when pre/h1 is tiny, so ordinary launches are untouched.
    f = _flatness_factor(p1, pre)
    if f < 1.0:
        rows_flat = []
        for row in rows_out:
            qs = row["q"]
            s = (f - 1.0) * math.log(max(qs[2], 1e-9) / p1)
            nq = [round(min(max(q * math.exp(s), 1.0), 1e9), 2) for q in qs]
            prev = 0.0
            clean = []
            for q in nq:
                if q < prev:
                    q = prev
                prev = q
                clean.append(q)
            rows_flat.append({"hour": row["hour"], "q": clean})
        rows_out = rows_flat

    return {"hours": rows_out}


def main(argv):
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        stem = name[:-3]
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        text = json.dumps(result, separators=(",", ":"))
        with open(os.path.join(out_dir, stem + ".out"), "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
