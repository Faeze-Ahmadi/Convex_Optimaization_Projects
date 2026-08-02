import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding="utf-8")

WAGE_SHOCK = 0.10  # 10% افزایش حقوق

PATH = Path(r"سوال 1/Input-Output-1400.xlsx")
if not PATH.exists():
    raise FileNotFoundError(f"File not found: {PATH.resolve()}")

s1 = pd.read_excel(PATH, sheet_name="1", header=None)  # جدول اصلی
s2 = pd.read_excel(PATH, sheet_name="2", header=None)  # ماتریس ضرایب مستقیم A
n = 74

sectors = s2.iloc[3:3+n, 0].astype(str).str.strip().tolist()
A = (
    s2.iloc[3:3+n, 2:2+n]
    .apply(pd.to_numeric, errors="coerce")
    .to_numpy(dtype=float)
)

if A.shape != (n, n):
    raise ValueError(f"A shape is {A.shape}, expected {(n, n)}")

nan_ratio = np.isnan(A).mean()
if nan_ratio > 0.01:
    print(
        f"Warning: A contains NaN values (ratio={nan_ratio:.3f}). Check the Excel block range.")
    A = np.nan_to_num(A, nan=0.0)

wage_row = 80
out_row = 85
wages = pd.to_numeric(s1.iloc[wage_row, 2:2+n],
                      errors="coerce").to_numpy(dtype=float)
x = pd.to_numeric(s1.iloc[out_row,  2:2+n],
                  errors="coerce").to_numpy(dtype=float)

if wages.shape[0] != n or x.shape[0] != n:
    raise ValueError("Length mismatch in wages or x vector.")

wages = np.nan_to_num(wages, nan=0.0)
x = np.nan_to_num(x, nan=0.0)

wage_unit = np.divide(wages, x, out=np.zeros_like(wages), where=(x != 0))
delta_v = WAGE_SHOCK * wage_unit

# مدل قیمتی لئونتیف, Δp = (I - A^T)^(-1) * Δv
M = np.eye(n) - A.T

condM = np.linalg.cond(M)
if condM > 1e10:
    print(
        f"Warning: (I - A^T) is ill-conditioned. cond={condM:.2e} -> results may be unstable.")

delta_p = np.linalg.solve(M, delta_v)

flag = True
if flag:
    print("Any NaN in delta_p?", np.isnan(delta_p).any())
    print("Any negative delta_p?", (delta_p < 0).any())

df = pd.DataFrame({
    "بخش": sectors,
    "سهم دستمزد از ستانده (w/x)": wage_unit,
    f"شوک مستقیم از {int(WAGE_SHOCK*100)}% افزایش حقوق (Δv)": delta_v,
    "تغییر قیمت تخمینی مدل (Δp)": delta_p
})

dv_col = f"شوک مستقیم از {int(WAGE_SHOCK*100)}% افزایش حقوق (Δv)"
df["ضریب تقویت شبکه (Δp/Δv)"] = np.divide(
    df["تغییر قیمت تخمینی مدل (Δp)"],
    df[dv_col].replace(0, np.nan)
)

df["Δp (%)"] = 100 * df["تغییر قیمت تخمینی مدل (Δp)"]

print("summary:")
print("mean Δp:", float(df["تغییر قیمت تخمینی مدل (Δp)"].mean()))
print("max  Δp:", float(df["تغییر قیمت تخمینی مدل (Δp)"].max()))
print("min  Δp:", float(df["تغییر قیمت تخمینی مدل (Δp)"].min()))

top10 = df.sort_values("تغییر قیمت تخمینی مدل (Δp)",
                       ascending=False).head(10).copy()
top10.insert(0, "rank", range(1, 11))

print("\ntop 10 (numbers only):")
print(top10[["rank", "Δp (%)"]].to_string(index=False))

OUT_DIR = Path(r"سوال 1")
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_xlsx = OUT_DIR / "leontief_price_shock_results.xlsx"
out_csv = OUT_DIR / "leontief_results_utf8.csv"
out_top = OUT_DIR / "top10_sectors.xlsx"
df.to_excel(out_xlsx, index=False)
df.to_csv(out_csv, index=False, encoding="utf-8-sig")
top10.to_excel(out_top, index=False)
print("\nsaved files in:", OUT_DIR.resolve())
print(out_xlsx.name)
print(out_csv.name)
print(out_top.name)
