[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Emjee22/gta6-twitch-ccu-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Emjee22/gta6-twitch-ccu-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# GTA VI Twitch CCU Forecast

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (19 Nov 2026, UTC), from everything recorded up to
the end of the first hour. For hours 2–24 it writes five quantiles
(5%, 25%, 50%, 75%, 95%) of that hour's concurrent-viewer count.

Entry for the make.fun bounty `gta6-twitch-ccu-forecast-01`.

## How it works

The program re-fits its model on every case, from that case's own
comparables (38 past big launches in the practice cases, 42 in the final).

The key finding from those launches: launch-day curves look wildly
different, but almost all of the difference is explained by one number —
how big hour 1 already is. A launch whose first UTC hour is nearly empty
(it unlocks late in the day) grows ten- or a hundred-fold by the evening;
a launch whose first hour is already huge (most of the world unlocked
before midnight UTC) barely grows at all. The median ratio to hour 1
therefore cannot be one fixed curve; it has to be conditioned on hour 1.

So for each hour `h`, the program fits across the comparable launches

```
log(viewers_h / viewers_1) ~ a_h + b_h * log(viewers_1)
                                 + c_h * log(pre-release baseline)
```

by kernel-weighted least squares, weighting each comparable by how close
its log hour-1 is to the target's (Gaussian kernel, bandwidth 1.5 in log
units). The second predictor is the category's own pre-release level: the
median hourly viewers in the 7 days before release day. The spread of the
five quantiles is the kernel-weighted residual standard deviation of that
fit (times 0.75), turned into normal quantiles.

Three guards keep the forecast sane where the data runs out: the predicted
log-ratio is clamped to the range the comparables show; the median never
falls below the 25th-percentile curve of the biggest launches (a launch
already near its peak in hour 1 does not shrink); and the log-median is
lightly smoothed across neighbouring hours.

Nothing is hard-coded from any single launch, and the program never sees
the answer: it only reads the case file it is given.

## Validation

Practice cases (answers = SullyGnome hourly averages for the launch day,
fetched from the public API named in the kit's `input/NOTICE.txt`; each
case's hour-1 value matches the case file exactly). Pinball loss on the
log scale, lower is better; the house baseline is the persistence
forecast from the kit (`q = hour-1 print`, spread widening over the day).

| case | launch | hour-1 | this entry | house baseline |
|------|--------|-------:|-----------:|---------------:|
| p00 | Marvel's Wolverine | 23,357 | **0.164** | 0.744 |
| p01 | Red Dead Redemption 2 | 116,550 | **0.206** | 0.287 |
| p02 | God of War Ragnarök | 269,517 | **0.139** | 0.215 |
| p03 | Marvel's Spider-Man 2 | 119,087 | **0.183** | 0.303 |
| **mean** | | | **0.173** | **0.387** |

Typical miss (median): 28% / 45% / 26% / 33% per case, vs 189% / 65% /
42% / 64% for the baseline. 90% coverage: 1.00 / 0.96 / 1.00 / 0.83.

Leave-one-out cross-validation over the 37 comparable launches (each
treated as a pseudo-case, scored against its own release day): pinball
loss **0.392** vs **0.930** for the baseline.

## Run

```sh
python3 main.py
```

Reads `$MAKEFUN_INPUT_DIR` (default `/input`), writes one
`$MAKEFUN_OUTPUT_DIR/<id>.out` per `<id>.in` found there, in name order.
Each output is `{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]},
...]}` for hours 2–24. Python 3.13 standard library only, no network,
deterministic (two runs write byte-identical output), about 0.25 s per
case.

## Files

- `main.py` — the forecaster (the whole entry)
- `README.md` — this file
- `LICENSE` — MIT

Data attributions: launch-day viewer counts used for validation are
SullyGnome.com hourly Twitch category averages, quoted with attribution
(see the kit's `input/NOTICE.txt`).
