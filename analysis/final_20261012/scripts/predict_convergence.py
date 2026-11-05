"""Can a configuration's convergence be predicted from its accuracy alone?

Leave-one-out: for each of the 14 configurations, fit the share of its wrong values that equal the
consensus error (convergence_by_config.csv) on task accuracy over the other 13 by least squares, predict the
held-out configuration, and report the error; then the same leaving out both configurations of the held-out model. Turns the accuracy-convergence association of Fig. 5a into an
out-of-sample test. Writes tables/predict_convergence.csv.
"""
import csv
import statistics

from common import TABLES, write_csv
from groups import MODEL, STUDIES

rows = [r for r in csv.DictReader(open(TABLES / "convergence_by_config.csv")) if r["study"] in STUDIES]
pts = [(r["study"], float(r["task_accuracy"]), float(r["wrong_equals_crowd"])) for r in rows]


def fit(sample):
    xs = [x for _, x, _ in sample]; ys = [y for _, _, y in sample]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


out = []; errs = {"configuration": [], "model": []}
for i, (s, x, y) in enumerate(pts):
    a, b = fit(pts[:i] + pts[i + 1:])                                   # leave one configuration out
    a_m, b_m = fit([q for q in pts if MODEL[q[0]] != MODEL[s]])         # leave both configurations of the model out
    pred, pred_m = a + b * x, a_m + b_m * x
    errs["configuration"].append(y - pred); errs["model"].append(y - pred_m)
    out.append([s, STUDIES[s][0] + (" T" if STUDIES[s][1] == "T" else " N"), round(x, 6), round(y, 6), round(pred, 6), round(y - pred, 6), round(pred_m, 6), round(y - pred_m, 6)])
    print(f"  {STUDIES[s][0]:18} {STUDIES[s][1]}  accuracy {x:.1%}  convergence {y:.1%}  predicted {pred:.1%} (model out {pred_m:.1%})  error {y - pred:+.1%} ({y - pred_m:+.1%})")
ys = [y for _, _, y in pts]; ss_tot = sum((y - statistics.mean(ys)) ** 2 for y in ys)
a, b = fit(pts)
for unit, e in errs.items():
    mae = statistics.mean(abs(x) for x in e); r2 = 1 - sum(x * x for x in e) / ss_tot
    print(f"leave-one-{unit}-out: mean absolute error {mae:.1%}, cross-validated R^2 {r2:.2f}")
    out.append([f"LOO_{unit.upper()}", "", "", "", round(mae, 6), round(r2, 6), "", ""])
print(f"full fit convergence = {a:+.3f} + {b:.3f} x accuracy ({b / 10:.1%} per 10 points of accuracy)")
out.append(["FULL_FIT", "", round(a, 6), round(b, 6), "", "", "", ""])
write_csv("predict_convergence.csv", ["study", "config", "task_accuracy", "convergence", "predicted_loo", "error_or_r2", "predicted_model_out", "error_model_out"], out)
