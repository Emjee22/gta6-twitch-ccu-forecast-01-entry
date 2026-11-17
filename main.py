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

This version therefore forecasts with three different views of the same
comparables and averages them in log space, equal thirds:

  1. Per-hour kernel-weighted regression, as in v1: for each hour h,
     log(viewers_h / viewers_1) ~ a + b*log(viewers_1)
     + c*log(pre-release 7-day median), weighted by how close the
     comparable's log hour-1 is to the target's (Gaussian kernel,
     bandwidth 1.5 in log units). The pre-release level is what saves
     small hour-1 launches: it separates "nobody was watching yet" from
     "nobody cares".

  2. Unlock-ramp regression: the same per-hour fit plus one more
     predictor, the number of hours since the launch's own US-East
     unlock (capped at 6). Each comparable's unlock hour is read off
     its own curve (its biggest jump in hours 4..9); the target's is
     computed from the calendar (US daylight-saving rule on day_start).
     This lets the step stay sharp instead of being smeared across
     neighbouring UTC hours.

  3. Pooled shape regression: one weighted least-squares fit over all
     (comparable, hour) pairs at once. The log-ratio is a linear
     combination of smooth basis functions of the hour -- a piecewise
     linear spline with knots at 3, 5, 9, 13, 17 and 21, plus a step
     and a post-step ramp at the launch's own unlock hour -- with
     coefficients that are themselves linear in (1, log hour-1,
     log pre-release). Thirty parameters over ~850 points instead of
     sixty-nine over 38, so the fitted curve is smooth by construction
     and the evening shape borrows strength from the morning.

The spread of each view is its own kernel-weighted residual standard
deviation (times 0.75, clamped to [0.25, 1.5] log units), turned into
normal quantiles. After averaging the three views' log-quantiles the
spread is rescaled by 0.8: on the practice launches the blended median
is good enough that the v1-width intervals are too wide, and 0.8 puts
the 90% coverage at 0.90, exactly the nominal level.

Three guards keep the forecast sane where the data runs out: each
view's predicted log-ratio is clamped to the range the comparables show
(plus half a log unit of slack); the median never falls below the
25th-percentile ratio curve of the biggest launches (a launch already
near its peak in hour 1 does not shrink); and the log-median is lightly
smoothed across neighbouring hours.

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
SPREAD_FACTOR = 0.75     # shrinks each view's residual spread
MIN_SPREAD = 0.25        # log units; keeps quantiles apart on degenerate fits
MAX_SPREAD = 1.5
PRE_WINDOW_DAYS = 7
# Blend weights for the three views (hourly OLS, unlock ramp, pooled
# shape) and the final spread rescale after blending.
BLEND_WEIGHTS = (0.34, 0.33, 0.33)
BLEND_SPREAD = 0.8
BIG_H1 = 200000.0        # a "big" launch, for the median floor
RAMP_CAP = 6             # hours of post-unlock ramp the model can see


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
    ramp. A jump has to clear 0.15 log units to count.
    """
    best_hour, best_jump = 6, 0.0
    for h in range(4, 10):
        jump = math.log(curve[h - 1] / curve[h - 2])
        if jump > best_jump:
            best_jump, best_hour = jump, h
    return best_hour if best_jump > 0.15 else 6


def _kernel_weight(log_h1, log_target):
    dist = (log_h1 - log_target) / KERNEL_BANDWIDTH
    return math.exp(-0.5 * dist * dist)


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


def _leg_hourly_ols(p1, pre, comps, use_ramp, step_target):
    """View 1 (use_ramp=False) and view 2 (use_ramp=True).

    One kernel-weighted fit per hour. The design is (1, log hour-1,
    log pre-release); the ramp view adds a fourth column, the hours
    since that launch's US-East unlock, capped at RAMP_CAP.
    Returns (median log-ratio per hour, residual sd per hour).
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
            x = (1.0, math.log(c["h1"]), math.log(max(c["pre"], 1.0)))
            if use_ramp:
                post = min(max(0, hour - c["step"]), RAMP_CAP)
                x = x + (float(post),)
            rows.append((_kernel_weight(math.log(c["h1"]), log_h1), x, lr))
        x_target = (1.0, log_h1, log_pre)
        if use_ramp:
            post_t = min(max(0, hour - step_target), RAMP_CAP)
            x_target = x_target + (float(post_t),)
        pred, sd, _ = _wls(rows, x_target)
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
    and curvature all shift with the size of the launch. The spread
    is the kernel-weighted residual sd at each hour around this fit.
    Returns (median log-ratio per hour, residual sd per hour).
    """
    log_h1 = math.log(p1)
    log_pre = math.log(max(pre, 1.0))
    n_basis = len(_shape_basis(2, 6))
    rows = []
    for c in comps:
        w = _kernel_weight(math.log(c["h1"]), log_h1)
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
            w = _kernel_weight(math.log(c["h1"]), log_h1)
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
    """Turn one view's (median, spread) per hour into forecast rows."""
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
    _smooth_medians(rows_out)
    return rows_out


def _smooth_medians(rows_out):
    """Lightly smooth the log-median across neighbouring hours.

    Each hour is fitted on its own, so the median can wiggle a little
    from one hour to the next. A (0.25, 0.5, 0.25) pass over the
    log-medians removes the wiggle; every quantile of an hour is
    shifted by the same log amount, so the spread is untouched.
    """
    medians = {row["hour"]: math.log(max(row["q"][2], 1e-9)) for row in rows_out}
    for row in rows_out:
        hour = row["hour"]
        total = 0.0
        weight_sum = 0.0
        for shift, weight in ((-1, 0.25), (0, 0.5), (1, 0.25)):
            if hour + shift in medians:
                total += weight * medians[hour + shift]
                weight_sum += weight
        shift = total / weight_sum - medians[hour]
        if shift == 0.0:
            continue
        factor = math.exp(shift)
        row["q"] = [round(min(max(q * factor, 1.0), 1e9), 2) for q in row["q"]]


def _blend_rows(leg_rows):
    """Average the three views' log-quantiles, then rescale the spread.

    Averaging in log space keeps the blend's median between the views'
    medians and its spread between their spreads; the 0.8 rescale
    around the blended median narrows the intervals to the width the
    blended median actually earns on the practice launches.
    """
    rows_out = []
    for hour in range(2, 25):
        by_hour = []
        for rows in leg_rows:
            by_hour.append(rows[hour - 2]["q"])
        blended = []
        for i in range(5):
            acc = 0.0
            for w, qs in zip(BLEND_WEIGHTS, by_hour):
                acc += w * math.log(max(qs[i], 1e-9))
            blended.append(acc)
        med_log = blended[2]
        final = []
        for v in blended:
            q = math.exp(med_log + (v - med_log) * BLEND_SPREAD)
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
        med1, spr1 = _leg_hourly_ols(p1, pre, comps, False, step_target)
        med2, spr2 = _leg_hourly_ols(p1, pre, comps, True, step_target)
        med3, spr3 = _leg_pooled_shape(p1, pre, comps, step_target)
        rows1 = _build_rows(p1, comps, med1, spr1)
        rows2 = _build_rows(p1, comps, med2, spr2)
        rows3 = _build_rows(p1, comps, med3, spr3)
        rows_out = _blend_rows([rows1, rows2, rows3])
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
