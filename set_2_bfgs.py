import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# تولید دیتاست مصنوعی برای تست الگوریتم‌ها
# یک دیتاست با 1000 نمونه و 10 ویژگی می‌ سازیم
X, y = make_regression(n_samples=1000, n_features=10,
                       noise=0.5, random_state=42)

# اضافه کردن ستون یک‌ها به X برای محاسبه عرض از مبدأ (Bias)
X = np.c_[np.ones(X.shape[0]), X]
n_samples, n_features = X.shape

# تعریف توابع پایه: تابع هزینه و گرادیان (کمترین مربعات)
def compute_loss(w, X_data, y_data):
    # محاسبه تابع هزینه کمترین مربعات (Least Squares)
    predictions = X_data.dot(w)
    return 0.5 * np.mean((predictions - y_data) ** 2)

def compute_gradient(w, X_data, y_data):
    # محاسبه گرادیان تابع هزینه 
    predictions = X_data.dot(w)
    return X_data.T.dot(predictions - y_data) / X_data.shape[0]

# پیاده‌سازی الگوریتم ب‌ف‌گ‌ش معمولی (استاندارد)
def standard_bfgs(X, y, max_iterations=50, lr=0.1):
    w = np.zeros(n_features)  # مقداردهی اولیه وزن‌ ها
    H = np.eye(n_features)   # مقداردهی اولیه ماتریس تقریب هسین 
    loss_history = []

    for i in range(max_iterations):
        # ثبت خطای کل در این تکرار
        loss_history.append(compute_loss(w, X, y))
        # محاسبه گرادیان روی کل داده‌ها
        g = compute_gradient(w, X, y)
        # تعیین جهت حرکت
        p = -H.dot(g)
        # به‌روزرسانی وزن‌ها
        w_new = w + lr * p
        # محاسبه گرادیان جدید برای یافتن yk
        g_new = compute_gradient(w_new, X, y)
        # بردارهای اختلاف
        s = w_new - w
        y_vec = g_new - g
        # به‌روزرسانی ماتریس هسین (فرمول ب‌ف‌گ‌ش)
        rho_den = np.dot(y_vec, s)
        if rho_den > 1e-10:  # جلوگیری از تقسیم بر صفر و حفظ مثبت معین بودن
            rho = 1.0 / rho_den
            I = np.eye(n_features)
            V = I - rho * np.outer(y_vec, s)
            H = V.T.dot(H).dot(V) + rho * np.outer(s, s)
        w = w_new
    return loss_history

# پیاده‌سازی الگوریتم O-BFGS نسخه برخط
def online_bfgs(X, y, epochs=50, batch_size=32, lr=0.1):
    w = np.zeros(n_features)
    H = np.eye(n_features)
    loss_history = []

    for epoch in range(epochs):
        # ثبت خطا در ابتدای هر ایپوک (برای مقایسه عادلانه با روش استاندارد)
        loss_history.append(compute_loss(w, X, y))
        # بر هم زدن تصادفی داده‌ ها در هر ایپوک
        indices = np.random.permutation(n_samples)
        # حرکت روی مینی‌ بچ‌ ها
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch, y_batch = X[batch_idx], y[batch_idx]

            # محاسبه گرادیان روی  مینی‌ بچ
            g = compute_gradient(w, X_batch, y_batch)
            p = -H.dot(g)
            # نرخ یادگیری کاهشی (در روش‌های آنلاین بهتر جواب می‌  دهد)
            step_size = lr / (1 + 0.05 * epoch)
            w_new = w + step_size * p
            #  محاسبه گرادیان جدید روی همان مینی‌بچ قبلی
            g_new_same_batch = compute_gradient(w_new, X_batch, y_batch)
            s = w_new - w
            y_vec = g_new_same_batch - g  # محاسبه y بدون نویزِ تغییر مینی‌  بچ
            # به‌روزرسانی ماتریس هسین
            rho_den = np.dot(y_vec, s)
            if rho_den > 1e-8:  # شرط انحنا
                rho = 1.0 / rho_den
                I = np.eye(n_features)
                V = I - rho * np.outer(y_vec, s)
                H = V.T.dot(H).dot(V) + rho * np.outer(s, s)
            w = w_new
    return loss_history

# اجرا و رسم نمودار مقایسه
print("running standard BFGS...")
bfgs_history = standard_bfgs(X, y, max_iterations=50)
print("running online BFGS (O-BFGS)...")
obfgs_history = online_bfgs(X, y, epochs=50, batch_size=32)
# تنظیمات اصلی فیگور و پس‌ مینه
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FFF0F5')   # رنگ پس‌ زمینه
# رسم خط BFGS استاندارد
plt.plot(bfgs_history, label='standard BFGS (full batch)',
         color='#FFB300', linewidth=2.5, markersize=10, markeredgecolor='#FF8F00')
# رسم خط O-BFGS
plt.plot(obfgs_history, label='online BFGS (mini batch)',
         color='#FF1493', linewidth=2.5, markersize=11, markeredgecolor='#C71585')
# استایل‌دهی به عنوان و محورها
plt.title('comparison of BFGS & O-BFGS \n(least squares convergence)',
          fontsize=15, fontweight='bold', color='#4A0E4E')
plt.xlabel('epochs / iterations', fontsize=12,
           fontweight='bold', color='#4A0E4E')
plt.ylabel('loss / MSE (log scale)', fontsize=12,
           fontweight='bold', color='#4A0E4E')
# تنظیم مقیاس و گرید لاین‌ها
plt.yscale('log')
plt.grid(True, which="both", linestyle="-.", color='#FFB6C1', alpha=0.7)
# استایل‌دهی به راهنمای نمودار
legend = plt.legend(fontsize=11, loc='upper right', frameon=True, shadow=True)
legend.get_frame().set_facecolor('#FFFFFF')
legend.get_frame().set_edgecolor('#FF1493')
# نمایش نهایی
plt.tight_layout()
plt.show()
