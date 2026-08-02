from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

paths = {
    "khodro": "مجموعه داده‌های سهام (سؤال دو)/داده‌های پردازش‌شده/خودرو_بازده.xlsx",
    "foolad": "مجموعه داده‌های سهام (سؤال دو)/داده‌های پردازش‌شده/فولاد_بازده.xlsx",
    "fmelli": "مجموعه داده‌های سهام (سؤال دو)/داده‌های پردازش‌شده/فملی_بازده.xlsx",
    "shepna": "مجموعه داده‌های سهام (سؤال دو)/داده‌های پردازش‌شده/شپنا_بازده.xlsx",
}

flag = False

dfs = {}
for k, p in paths.items():
    df = pd.read_excel(p)
    if flag:
        print("\n", k, "columns:", [repr(c) for c in df.columns])
    dfs[k] = df


def standardize_two_cols(df, name_for_return: str):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    date_candidates = [c for c in df.columns if c.lower() in (
        "dtyyyymmdd", "date", "dt", "تاریخ")]
    if not date_candidates:
        date_col = df.columns[0]
    else:
        date_col = date_candidates[0]

    ret_candidates = [c for c in df.columns if c.lower() in (
        "return", "بازده", name_for_return.lower())]
    if not ret_candidates:
        ret_col = df.columns[1] if len(df.columns) > 1 else None
    else:
        ret_col = ret_candidates[0]

    if ret_col is None:
        raise ValueError(
            "The output column was not found. The file must have at least two columns.")

    out = df[[date_col, ret_col]].rename(
        columns={date_col: "Date", ret_col: name_for_return})

    out["Date"] = pd.to_numeric(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).astype({"Date": "int64"})
    out = out.sort_values("Date").drop_duplicates("Date")

    out[name_for_return] = pd.to_numeric(out[name_for_return], errors="coerce")

    return out.dropna(subset=[name_for_return])


khodro = standardize_two_cols(dfs["khodro"], "Khodro")
foolad = standardize_two_cols(dfs["foolad"], "Foolad")
fmelli = standardize_two_cols(dfs["fmelli"], "Fmelli")
shepna = standardize_two_cols(dfs["shepna"], "Shepna")

df = khodro.merge(foolad, on="Date", how="inner") \
           .merge(fmelli, on="Date", how="inner") \
           .merge(shepna, on="Date", how="inner")

if flag:
    print("Final shape:", df.shape)
    print(df.head())
    print(df.tail())

df.to_csv("مجموعه داده‌های سهام (سؤال دو)/بازده_بازار_چهارسهم.csv",
          index=False, encoding="utf-8-sig")
df.to_excel(
    "مجموعه داده‌های سهام (سؤال دو)/بازده_بازار_چهارسهم.xlsx", index=False)
if flag:
    print("Saved.")

X = df[["Khodro", "Foolad", "Fmelli", "Shepna"]].copy()

Xc = X - X.mean(axis=0)
Sigma = np.cov(Xc.values, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(Sigma)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

explained_ratio_eig = eigvals / eigvals.sum()

if flag:
    print("Eigenvalues:", eigvals)
    print("Explained variance ratio (Eigen):", explained_ratio_eig)
    print("First eigenvector (factor 1):", eigvecs[:, 0])

pca = PCA()
pca.fit(Xc)

if flag:
    print("PCA explained variance ratio:", pca.explained_variance_ratio_)
    print("PCA first component:", pca.components_[0])

raw_khodro = pd.read_csv(
    "مجموعه داده‌های سهام (سؤال دو)/داده‌های خام/خودرو_خام.csv"
)

raw_khodro.columns = [c.strip() for c in raw_khodro.columns]

price_df = raw_khodro[["<DTYYYYMMDD>", "<CLOSE>"]].copy()
price_df.columns = ["Date", "Close"]

price_df["Date"] = pd.to_numeric(price_df["Date"], errors="coerce")
price_df = price_df.dropna(subset=["Date"])
price_df["Date"] = price_df["Date"].astype("int64")

price_df = price_df.sort_values("Date").drop_duplicates("Date")

if flag:
    print(price_df.head())

df_full = df.merge(price_df, on="Date", how="inner")

if flag:
    print("Merged shape:", df_full.shape)
    print(df_full.head())

df_full["Khodro_next"] = df_full["Khodro"].shift(-1)
df_full["Close_next_real"] = df_full["Close"].shift(-1)

df_model = df_full.dropna().copy()

features = ["Khodro", "Foolad", "Fmelli", "Shepna"]

X_all = df_model[features].values
y_all = df_model["Khodro_next"].values

split_ratio = 0.8
n = len(df_model)
split = int(split_ratio * n)

X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]

close_test = df_model["Close"].values[split:]
close_next_test_real = df_model["Close_next_real"].values[split:]
date_test = df_model["Date"].values[split:]

print("\nTrain size:", len(X_train), "| Test size:", len(X_test))


def run_pca_regression(k: int):
    pca = PCA(n_components=k)
    pca.fit(X_train)

    Z_train = pca.transform(X_train)
    Z_test = pca.transform(X_test)

    A_train = np.column_stack([np.ones(len(Z_train)), Z_train])
    A_test = np.column_stack([np.ones(len(Z_test)), Z_test])

    theta = np.linalg.lstsq(A_train, y_train, rcond=None)[0]

    y_pred_test = A_test @ theta
    price_pred_test = close_test * (1 + y_pred_test)

    mse = mean_squared_error(close_next_test_real, price_pred_test)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(
        close_next_test_real, price_pred_test)

    out = pd.DataFrame({
        "Date": date_test,
        "Close_t": close_test,
        "Close_next_real": close_next_test_real,
        "Return_pred": y_pred_test,
        "Price_pred": price_pred_test
    })
    out.to_excel(f"prediction_khodro_pca_k{k}.xlsx", index=False)

    print(f"\nresults (k={k}) ")
    print("Z_train shape:", Z_train.shape, "| Z_test shape:", Z_test.shape)
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("theta =", theta, "| number of coeffs =", len(theta))
    print("TEST MSE :", mse)
    print("TEST RMSE:", rmse)
    print("TEST MAPE:", mape)
    print(f"Saved: prediction_khodro_pca_k{k}.xlsx")

    return rmse, mape, theta


rmse1, mape1, theta1 = run_pca_regression(1)
rmse2, mape2, theta2 = run_pca_regression(2)

print("\ncomparison")
print(f"k=1 -> RMSE={rmse1:.3f} | MAPE={mape1:.5f}")
print(f"k=2 -> RMSE={rmse2:.3f} | MAPE={mape2:.5f}")

df_plot = pd.read_excel("prediction_khodro_pca_k1.xlsx")

plt.figure()
plt.plot(df_plot["Close_next_real"].values, label="Real Price (t+1)")
plt.plot(df_plot["Price_pred"].values, label="Predicted Price (t+1)")
plt.legend()
plt.title("Khodro Price Prediction using PCA (k=1)")
plt.xlabel("Test Sample Index")
plt.ylabel("Price")
plt.show()
