# Stability-aware adaptive RK4 experiment

This directory is a standalone experiment. It does not import project code,
read previous numerical tables, or modify the report. Both controllers are
implemented in `stability_aware_adaptive_rk4.py` and differ only by the local
stability cap.

## Controller definition

The baseline uses RK4 step doubling, the accepted two-half-step state, the
Richardson divisor 15, and the component-scaled indicator

```text
eta = max_i |(y_fine - y_whole)_i / 15| / (atol_ratio_i + |y_i|).
```

The stability-aware variant additionally applies

```text
H_trial <= 0.9 h_stab,
```

where `h_stab` is found by bisection from the two eigenvalues of the reduced
Robertson Jacobian. It is the largest outer step `H` up to `h_max` satisfying

```text
max_j |R((H/2) lambda_j)| <= 1,
R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24.
```

All other parameters are identical: `h0=1e-6`, `h_min=1e-12`, `h_max=5e-2`,
error safety `0.9`, growth limit `5`, shrink limit `0.2`, and
`atol_ratio=(1e-10,1e-13,1e-10)`.

## Raw comparison

| Controller | tol | reached 40 | accepted | rejected | RHS calls | eig/J evals | E_global | E_end | min y | y2(40) | max H | cap-active accepted | min/median/max H/h_stab |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| standard | 1e-10 | yes | 31262 | 1 | 375156 | 0 | 7.433e-14 | 2.764e-16 | 0 | 9.185534764834e-6 | 1.480e-3 | 0% | 1.426e-5 / 0.6928 / 0.9008 |
| stability-aware | 1e-10 | yes | 31262 | 1 | 375156 | 62524 | 7.433e-14 | 2.132e-16 | 0 | 9.185534764771e-6 | 1.480e-3 | 0.365% | 1.426e-5 / 0.6928 / 0.9000 |
| standard | 1e-8 | yes | 22056 | 8309 | 364380 | 0 | 2.138e-12 | 2.069e-15 | 0 | 9.185534766627e-6 | 3.362e-3 | 0% | 2.000e-5 / 0.8509 / 1.9976 |
| stability-aware | 1e-8 | yes | 23046 | 0 | 276552 | 46091 | 2.138e-12 | 8.176e-17 | 0 | 9.185534764640e-6 | 2.187e-3 | 91.087% | 2.000e-5 / 0.9000 / 0.9000 |
| standard | 1e-6 | yes | 20764 | 9354 | 361416 | 0 | 2.955e-10 | 4.606e-11 | 0 | 9.185580824122e-6 | 4.885e-3 | 0% | 2.000e-5 / 0.9304 / 2.0015 |
| stability-aware | 1e-6 | yes | 22830 | 0 | 273960 | 45659 | 4.053e-11 | 3.496e-17 | 0 | 9.185534764593e-6 | 2.302e-3 | 99.847% | 2.000e-5 / 0.9000 / 0.9000 |
| standard | 1e-4 | yes | 20684 | 8444 | 349536 | 0 | 1.981e-6 | 2.060e-10 | 0 | 9.185328783439e-6 | 5.009e-3 | 0% | 2.000e-5 / 0.9328 / 2.0579 |
| stability-aware | 1e-4 | yes | 22814 | 0 | 273768 | 45627 | 1.364e-9 | 6.598e-17 | 0 | 9.185534764624e-6 | 2.302e-3 | 99.921% | 2.000e-5 / 0.9000 / 0.9000 |

The complete machine-readable table is
`output/raw/stability_aware_comparison.csv`. Per-run trajectories, steps, local
stability caps, and cap flags are stored in the eight NPZ files.

## Research questions

**Q1. Did the error controller propose steps beyond the local stability scale?**
Yes. At tolerances `1e-8`, `1e-6`, and `1e-4`, the standard controller accepted
6459, 8524, and 8666 outer steps with `H/h_stab > 1`. Its maximum ratios were
1.998, 2.002, and 2.058. At `1e-10`, no accepted step exceeded `h_stab`.

**Q2. Did the cap reduce rejection, instability, negativity, or large error?**
Both methods reached `t=40` without negative concentrations. The cap reduced
rejections from 8309, 9354, and 8444 to zero at the three looser tolerances. It
reduced global error by a factor of about 7.29 at `1e-6` and 1452 at `1e-4`.
At `1e-8`, the reported global errors are equal at the common-grid error floor.

**Q3. Did it improve computational efficiency?**
Measured by RHS calls, yes at the three looser tolerances: the reductions were
24.10%, 24.20%, and 21.68%. The number of accepted steps increased by 4.49%,
9.95%, and 10.30%, but avoiding rejected step-doubling trials more than offset
that increase. This comparison does not include a wall-clock claim. The new
controller also required 46091--62524 reduced-Jacobian eigenvalue evaluations,
which are absent from the baseline algorithm.

**Q4. At which tolerances was the cap active?**
It affected 0.365% of accepted steps at `1e-10`, 91.087% at `1e-8`, 99.847% at
`1e-6`, and 99.921% at `1e-4`.

**Q5. Was the cap only overhead because mixed control was already stable?**
Only at `1e-10`: both methods had the same accepted/rejected counts, RHS work,
and global error, so the extra spectral calculations produced no practical
benefit. At the looser tolerances, the cap changed the trajectory of step-size
selection, removed thousands of rejections, and reduced RHS work. Its net value
therefore depends on whether the cost of the small 2-by-2 eigenvalue calculation
is significant in the target implementation.

## Reproduction

```bash
python3 -m unittest -v test_stability_aware_adaptive_rk4.py
python3 stability_aware_adaptive_rk4.py
```

The tests verify the reduced Jacobian, the RK4 stability boundary, and the
enforcement of the `0.9 h_stab` cap.
