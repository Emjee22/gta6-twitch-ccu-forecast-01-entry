"""Forecast launch-day Twitch viewers, hour by hour, from hour 1.

Reads one case file per run (schema makefun/twitch-ccu-case/v1) and writes,
for hours 2..24, five quantiles of that hour's concurrent-viewer count.

Method (fitted fresh on every case, from that case's own comparables):
  Launch-day curves differ wildly in shape, but almost all of the difference
  is explained by one number: how big hour 1 already is. A launch whose
  first UTC hour is nearly empty (it unlocks late in the day) grows ten- or
  a hundred-fold by the evening; a launch whose first hour is already huge
  (most of the world unlocked before midnight UTC) barely grows at all.
  So for each hour h I fit, across the comparable launches,

      log(viewers_h / viewers_1)  ~  a_h + b_h * log(viewers_1)
                                        + c_h * log(pre-release baseline)

  by kernel-weighted least squares, weighting each comparable by how close
  its log hour-1 is to the target's (Gaussian kernel, bandwidth 1.5 in log
  units). The second predictor is the category's own pre-release level:
  the median hourly viewers in the 7 days before release day, which says
  how much of the launch's audience was already watching before it began.
  The spread of the five quantiles is the kernel-weighted residual
  standard deviation of that fit (times 0.75, tuned on practice launches),
  turned into normal quantiles. Nothing is hard-coded from any one launch:
  the whole curve is re-estimated from whatever comparables the case holds.

  Three guards keep the forecast sane where the data runs out: the
  predicted log-ratio is clamped to the range the comparables show; the
  median never falls below the 25th-percentile curve of the biggest
  launches (a launch already near its peak in hour 1 does not shrink);
  and the log-median is lightly smoothed across neighbouring hours.

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
SPREAD_FACTOR = 0.75     # shrinks the residual spread to pinball-optimal width
MIN_SPREAD = 0.25        # log units; keeps quantiles apart on degenerate fits
MAX_SPREAD = 1.5
PRE_WINDOW_DAYS = 7


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


def _fit_hour(rows, log_h1, log_pre):
    """Kernel-weighted least squares for one hour.

    rows: (weight, log hour-1, log pre-release median, log ratio).
    Returns (predicted log ratio, weighted residual sd).
    """
    size = 3
    mat = [[0.0] * size for _ in range(size)]
    rhs = [0.0] * size
    for w, x, z, y in rows:
        vec = (1.0, x, z)
        for i in range(size):
            rhs[i] += w * vec[i] * y
            for j in range(size):
                mat[i][j] += w * vec[i] * vec[j]
    coef = _solve(mat, rhs)
    if coef is None:
        return 0.0, 1.0
    pred = coef[0] + coef[1] * log_h1 + coef[2] * log_pre
    weight_sum = sum(w for w, _, _, _ in rows)
    var = 0.0
    for w, x, z, y in rows:
        resid = y - (coef[0] + coef[1] * x + coef[2] * z)
        var += w * resid * resid
    sd = math.sqrt(var / weight_sum) if weight_sum > 0 else 1.0
    return pred, sd


def _smooth_medians(rows_out):
    """Lightly smooth the log-median across neighbouring hours.

    Each hour is fitted on its own, so the median can wiggle a little from
    one hour to the next. A (0.25, 0.5, 0.25) pass over the log-medians
    removes the wiggle; every quantile of an hour is shifted by the same
    log amount, so the spread is untouched.
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
        comps.append((float(curve[0]), c_pre, curve))

    log_h1 = math.log(p1)
    log_pre = math.log(max(pre, 1.0))
    rows_out = []

    if len(comps) >= 8:
        # A launch already near its peak in hour 1 cannot be predicted to
        # shrink far below it: the 25th-percentile ratio of the biggest
        # comparable launches is a floor for the median (and, scaled, for
        # the lower quantiles). It only ever binds for very large hour-1
        # counts, where the fitted line would otherwise extrapolate down.
        big = [c for c in comps if c[0] >= 200000.0]
        floor = {}
        if len(big) >= 3:
            for hour in range(2, 25):
                floor[hour] = _quantile(
                    [math.log(c[2][hour - 1] / c[0]) for c in big], 0.25)

        fitted = {}
        for hour in range(2, 25):
            rows = []
            log_ratios = []
            for h1, c_pre, curve in comps:
                lx = math.log(h1)
                dist = (lx - log_h1) / KERNEL_BANDWIDTH
                weight = math.exp(-0.5 * dist * dist)
                lr = math.log(curve[hour - 1] / h1)
                log_ratios.append(lr)
                rows.append((weight, lx, math.log(max(c_pre, 1.0)), lr))
            pred, sd = _fit_hour(rows, log_h1, log_pre)
            # Never extrapolate the log-ratio outside the range the
            # comparables actually show (plus half a log unit of slack).
            pred = min(max(pred, min(log_ratios) - 0.5),
                       max(log_ratios) + 0.5)
            fitted[hour] = (pred, sd)

        for hour in range(2, 25):
            pred, sd = fitted[hour]
            spread = min(max(sd * SPREAD_FACTOR, MIN_SPREAD), MAX_SPREAD)
            quants = [p1 * math.exp(pred + z * spread) for z in Z_VALUES]
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
    else:
        # Too few usable comparables: fall back to the plain empirical
        # quantiles of the log-ratios, or flat persistence if even that
        # is impossible. This path is only a safety net.
        for hour in range(2, 25):
            ratios = [math.log(c[2][hour - 1] / c[0]) for c in comps]
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
