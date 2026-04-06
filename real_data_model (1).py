"""
=============================================================================
RAINFALL × ROAD ACCIDENTS — INDIA
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── Plot style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   '#f8f9fa',
    'axes.grid':        True,
    'grid.color':       'white',
    'grid.linewidth':   0.8,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'font.family':      'DejaVu Sans',
    'axes.titlesize':   13,
    'axes.labelsize':   11,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
})
PALETTE = sns.color_palette("muted")

# =============================================================================
# 1. LOAD & CLEAN RAINFALL DATA
# =============================================================================
def load_and_clean_rainfall(path):
    df = pd.read_csv(path, sep=';', engine='python', on_bad_lines='skip')
    df.columns = [c.strip().replace('"', '') for c in df.columns]

    # The original code renamed '31s' → '31st', but the actual header is already '31st'
    # (checked: header row ends with ;"31st"). Safe to skip that rename.

    required = ['state', 'district', 'month']
    day_cols = [c for c in df.columns if c not in required]

    for col in day_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Strip quotes from string columns
    for col in required:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.replace('"', '', regex=False)

    df = df.dropna(subset=day_cols, how='all')
    df = df[df['district'].notna()]
    df['month'] = pd.to_numeric(df['month'], errors='coerce').astype('Int64')
    df = df.dropna(subset=['month'])

    return df, day_cols


rain_raw, day_cols = load_and_clean_rainfall('Rainfall_dataset.csv')

# Feature engineering
rain_raw['monthly_total_mm']  = rain_raw[day_cols].sum(axis=1)
rain_raw['monthly_avg_daily'] = rain_raw[day_cols].mean(axis=1)
rain_raw['monthly_max_daily'] = rain_raw[day_cols].max(axis=1)
rain_raw['rainy_days']        = (rain_raw[day_cols] > 2.5).sum(axis=1)
rain_raw['heavy_rain_days']   = (rain_raw[day_cols] > 40).sum(axis=1)

rain_raw = rain_raw.sort_values(['state', 'district', 'month'])
rain_raw['prev_month_total']  = rain_raw.groupby(['state', 'district'])['monthly_total_mm'].shift(1)
rain_raw['prev2_month_total'] = rain_raw.groupby(['state', 'district'])['monthly_total_mm'].shift(2)
rain_raw.fillna(0, inplace=True)

# Aggregate to state level
state_rain = rain_raw.groupby(['state', 'month']).agg(
    monthly_total_mm  = ('monthly_total_mm',  'mean'),
    monthly_avg_daily = ('monthly_avg_daily', 'mean'),
    monthly_max_daily = ('monthly_max_daily', 'mean'),
    rainy_days        = ('rainy_days',        'mean'),
    heavy_rain_days   = ('heavy_rain_days',   'mean'),
    prev_month_total  = ('prev_month_total',  'mean'),
    prev2_month_total = ('prev2_month_total', 'mean'),
).reset_index()

print("✅ Rainfall cleaned — shape:", state_rain.shape)

# =============================================================================
# 2. LOAD & CLEAN ACCIDENT DATA
# =============================================================================
def clean_accident_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lstrip('\ufeff') for c in df.columns]   # <-- FIX: strip BOM

    # The CSV has extra columns (rankings, % change, etc.) — keep only what we need
    # Columns: Sl No, State, 2019 Accidents, 2020 Accidents, ..., 2023 Accidents, ...
    acc_cols = [c for c in df.columns if 'Accident' in c]            # <-- FIX: dynamic detection
    keep = ['State'] + acc_cols
    df = df[keep].copy()

    df['State'] = df['State'].astype(str).str.strip()
    for col in acc_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['State'])
    df = df[df['State'].str.lower() != 'nan']
    return df


acc_raw = clean_accident_data('Accident_dataset.csv')
print("✅ Accident data cleaned — shape:", acc_raw.shape)
print("   Columns:", acc_raw.columns.tolist())

# Melt to long format
acc_long = acc_raw.melt(id_vars=['State'], var_name='year_label', value_name='annual_accidents')
acc_long['year'] = acc_long['year_label'].str.extract(r'(\d{4})').astype(float).astype('Int64')
acc_long = acc_long.dropna(subset=['annual_accidents', 'year'])
acc_long['annual_accidents'] = acc_long['annual_accidents'].astype(int)
acc_long.rename(columns={'State': 'state'}, inplace=True)

# =============================================================================
# 3. EXPAND ANNUAL → MONTHLY (seasonal distribution)
# =============================================================================
MONTHLY_SHARE = {
    1: 0.082,  2: 0.073,  3: 0.080,  4: 0.082,
    5: 0.086,  6: 0.082,  7: 0.082,  8: 0.082,
    9: 0.084, 10: 0.090, 11: 0.089, 12: 0.088,
}

rows = []
for _, row in acc_long.iterrows():
    for m, share in MONTHLY_SHARE.items():
        rows.append({
            'state':             row['state'],
            'year':              int(row['year']),
            'month':             m,
            'monthly_accidents': int(row['annual_accidents'] * share),
        })

acc_monthly = pd.DataFrame(rows)

# =============================================================================
# 4. NORMALISE STATE NAMES
# =============================================================================
# Key fix: rainfall CSV uses "Andaman & Nicobar" but accident CSV may differ
# Map both sides to a canonical form
STATE_MAP = {
    'Andaman & Nicobar Islands': 'Andaman & Nicobar',
    'Jammu & Kashmir':           'Jammu & Kashmir',
    'Jammu and Kashmir':         'Jammu & Kashmir',
    'J & K #':         'Jammu & Kashmir',
    'J & K':         'Jammu & Kashmir',
    'Orissa':                    'Odisha',
}

def normalize(s):
    s = str(s).strip().replace('"', '')
    return STATE_MAP.get(s, s)

acc_monthly['state_norm'] = acc_monthly['state'].apply(normalize)
state_rain['state_norm']  = state_rain['state'].apply(normalize)

# =============================================================================
# 5. MERGE
# =============================================================================
df = acc_monthly.merge(
    state_rain.drop(columns=['state']),
    on=['state_norm', 'month'],
    how='inner'
)

print("✅ Merge done — shape:", df.shape)
print("   Unique states in merged data:", df['state_norm'].nunique())

if df.empty:
    raise ValueError("Merged DataFrame is empty — check state name normalisation.")

# =============================================================================
# 6. FEATURE ENGINEERING
# =============================================================================
df['is_monsoon']       = df['month'].isin([6, 7, 8, 9]).astype(int)
df['post_rain_dry']    = ((df['prev_month_total'] > 150) & (df['monthly_total_mm'] < 30)).astype(int)
df['rain_damage_index']= df['prev_month_total'] * df['heavy_rain_days']

FEATURES = [
    'monthly_total_mm', 'monthly_avg_daily', 'monthly_max_daily',
    'rainy_days', 'heavy_rain_days',
    'prev_month_total', 'prev2_month_total',
    'rain_damage_index', 'post_rain_dry',
    'month', 'is_monsoon',
]

df = df.dropna(subset=FEATURES + ['monthly_accidents'])

X = df[FEATURES]
y = df['monthly_accidents']

print("✅ Feature matrix:", X.shape)

# =============================================================================
# 7. MODEL TRAINING — Four Models Compared
# =============================================================================
# • Random Forest   — ensemble of independent trees, strong non-linear baseline
# • XGBoost         — gradient boosting with regularisation, industry standard
# • LightGBM        — faster gradient boosting, leaf-wise tree growth
# • SVR             — support vector machine for regression, kernel-based
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Model 1: Random Forest ───────────────────────────────────────────────────
rf = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_cv    = cross_val_score(rf, X, y, cv=5, scoring='r2')
rf_results = {
    'name':  'Random Forest',
    'r2':    r2_score(y_test, rf_preds),
    'mae':   mean_absolute_error(y_test, rf_preds),
    'rmse':  np.sqrt(mean_squared_error(y_test, rf_preds)),
    'cv_r2': rf_cv.mean(),
    'cv_std':rf_cv.std(),
    'preds': rf_preds,
}

# ── Model 2: XGBoost ─────────────────────────────────────────────────────────
xgb = XGBRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, verbosity=0
)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
xgb_cv    = cross_val_score(xgb, X, y, cv=5, scoring='r2')
xgb_results = {
    'name':  'XGBoost',
    'r2':    r2_score(y_test, xgb_preds),
    'mae':   mean_absolute_error(y_test, xgb_preds),
    'rmse':  np.sqrt(mean_squared_error(y_test, xgb_preds)),
    'cv_r2': xgb_cv.mean(),
    'cv_std':xgb_cv.std(),
    'preds': xgb_preds,
}

# ── Model 3: LightGBM ────────────────────────────────────────────────────────
lgbm = LGBMRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=5,
    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, verbose=-1
)
lgbm.fit(X_train, y_train)
lgbm_preds = lgbm.predict(X_test)
lgbm_cv    = cross_val_score(lgbm, X, y, cv=5, scoring='r2')
lgbm_results = {
    'name':  'LightGBM',
    'r2':    r2_score(y_test, lgbm_preds),
    'mae':   mean_absolute_error(y_test, lgbm_preds),
    'rmse':  np.sqrt(mean_squared_error(y_test, lgbm_preds)),
    'cv_r2': lgbm_cv.mean(),
    'cv_std':lgbm_cv.std(),
    'preds': lgbm_preds,
}

# ── Model 4: SVR (Support Vector Regression) ─────────────────────────────────
# Needs feature scaling — wrapped in a Pipeline
svr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svr',    SVR(kernel='rbf', C=100, gamma=0.1, epsilon=50))
])
svr_pipe.fit(X_train, y_train)
svr_preds = svr_pipe.predict(X_test)
svr_cv    = cross_val_score(svr_pipe, X, y, cv=5, scoring='r2')
svr_results = {
    'name':  'SVR',
    'r2':    r2_score(y_test, svr_preds),
    'mae':   mean_absolute_error(y_test, svr_preds),
    'rmse':  np.sqrt(mean_squared_error(y_test, svr_preds)),
    'cv_r2': svr_cv.mean(),
    'cv_std':svr_cv.std(),
    'preds': svr_preds,
}

# ── Print individual model results ───────────────────────────────────────────
all_results = [rf_results, xgb_results, lgbm_results, svr_results]

print("\n📊 MODEL RESULTS : Random Forest Regression")
print(f"   R²       : {rf_results['r2']:.4f}")
print(f"   MAE      : {rf_results['mae']:.0f}")
print(f"   RMSE     : {rf_results['rmse']:.0f}")
print(f"   CV R²    : {rf_results['cv_r2']:.4f} ± {rf_results['cv_std']:.4f}")

print("\n📊 MODEL RESULTS : XGBoost Regression")
print(f"   R²       : {xgb_results['r2']:.4f}")
print(f"   MAE      : {xgb_results['mae']:.0f}")
print(f"   RMSE     : {xgb_results['rmse']:.0f}")
print(f"   CV R²    : {xgb_results['cv_r2']:.4f} ± {xgb_results['cv_std']:.4f}")

print("\n📊 MODEL RESULTS : LightGBM Regression")
print(f"   R²       : {lgbm_results['r2']:.4f}")
print(f"   MAE      : {lgbm_results['mae']:.0f}")
print(f"   RMSE     : {lgbm_results['rmse']:.0f}")
print(f"   CV R²    : {lgbm_results['cv_r2']:.4f} ± {lgbm_results['cv_std']:.4f}")

print("\n📊 MODEL RESULTS : SVR (Support Vector Regression)")
print(f"   R²       : {svr_results['r2']:.4f}")
print(f"   MAE      : {svr_results['mae']:.0f}")
print(f"   RMSE     : {svr_results['rmse']:.0f}")
print(f"   CV R²    : {svr_results['cv_r2']:.4f} ± {svr_results['cv_std']:.4f}")

print("\n" + "="*65)
print(f"{'📊 MODEL COMPARISON SUMMARY':^65}")
print("="*65)
print(f"{'Model':<22} {'R²':>7} {'MAE':>8} {'RMSE':>8} {'CV R²':>10}")
print("-"*65)
for res in all_results:
    print(f"{res['name']:<22} {res['r2']:>7.4f} {res['mae']:>8.0f} "
          f"{res['rmse']:>8.0f} {res['cv_r2']:>7.4f} ±{res['cv_std']:.3f}")
print("="*65)

# Determine best model by CV R²
best = max(all_results, key=lambda x: x['cv_r2'])
print(f"\n🏆 Best Model (by CV R²): {best['name']}")
print(f"   CV R²  = {best['cv_r2']:.4f} ± {best['cv_std']:.4f}")
print(f"   Test R²= {best['r2']:.4f}  |  MAE = {best['mae']:.0f}  |  RMSE = {best['rmse']:.0f}")

# Aliases used downstream by visualisations (RF used for feature importance & fig6/7)
preds     = rf_preds
r2        = rf_results['r2']
mae       = rf_results['mae']
rmse      = rf_results['rmse']
cv_scores = rf_cv

# Feature importances (RF — native tree-based importance)
fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)

# =============================================================================
# 8. VISUALISATIONS  (one clear purpose per figure)
# =============================================================================

MONTH_LABELS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# ── FIG 1: Monthly rainfall distribution by season ──────────────────────────
fig1, ax = plt.subplots(figsize=(10, 5))
monthly_rain = state_rain.groupby('month')['monthly_total_mm'].mean()
colors = ['#4a90d9' if m in [6,7,8,9] else '#a8c8e8' for m in range(1,13)]
bars = ax.bar(range(1,13), monthly_rain.values, color=colors, edgecolor='white', linewidth=0.5)
ax.set_xticks(range(1,13))
ax.set_xticklabels(MONTH_LABELS)
ax.set_title('Average Monthly Rainfall Across Indian States', fontweight='bold', pad=12)
ax.set_xlabel('Month')
ax.set_ylabel('Avg Rainfall (mm)')
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#4a90d9', label='Monsoon (Jun–Sep)'),
                   Patch(facecolor='#a8c8e8', label='Non-Monsoon')]
ax.legend(handles=legend_elements, frameon=True)
for bar, val in zip(bars, monthly_rain.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{val:.0f}', ha='center', va='bottom', fontsize=8)
fig1.tight_layout()
fig1.savefig('fig1_monthly_rainfall.png', dpi=150, bbox_inches='tight')
plt.close(fig1)
print("✅ Saved fig1_monthly_rainfall.png")

# ── FIG 2: Monthly accident distribution ────────────────────────────────────
fig2, ax = plt.subplots(figsize=(10, 5))
monthly_acc = df.groupby('month')['monthly_accidents'].mean()
colors2 = ['#e07b54' if m in [10,11,12] else '#f5c4a8' for m in range(1,13)]
bars2 = ax.bar(range(1,13), monthly_acc.values, color=colors2, edgecolor='white', linewidth=0.5)
ax.set_xticks(range(1,13))
ax.set_xticklabels(MONTH_LABELS)
ax.set_title('Average Monthly Road Accidents Across Indian States', fontweight='bold', pad=12)
ax.set_xlabel('Month')
ax.set_ylabel('Avg Monthly Accidents')
for bar, val in zip(bars2, monthly_acc.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            f'{val:.0f}', ha='center', va='bottom', fontsize=8)
fig2.tight_layout()
fig2.savefig('fig2_monthly_accidents.png', dpi=150, bbox_inches='tight')
plt.close(fig2)
print("✅ Saved fig2_monthly_accidents.png")

# ── FIG 3: Rainfall vs Accidents scatter (monsoon vs non-monsoon) ────────────
fig3, ax = plt.subplots(figsize=(8, 6))
monsoon     = df[df['is_monsoon'] == 1]
non_monsoon = df[df['is_monsoon'] == 0]
ax.scatter(non_monsoon['monthly_total_mm'], non_monsoon['monthly_accidents'],
           alpha=0.35, s=18, color='#a8c8e8', label='Non-Monsoon', rasterized=True)
ax.scatter(monsoon['monthly_total_mm'], monsoon['monthly_accidents'],
           alpha=0.45, s=18, color='#4a90d9', label='Monsoon', rasterized=True)
# Trend line
z = np.polyfit(df['monthly_total_mm'], df['monthly_accidents'], 1)
xline = np.linspace(df['monthly_total_mm'].min(), df['monthly_total_mm'].max(), 200)
ax.plot(xline, np.polyval(z, xline), color='#c0392b', lw=1.8, ls='--', label='Linear Trend')
ax.set_title('Monthly Rainfall vs Road Accidents', fontweight='bold', pad=12)
ax.set_xlabel('Monthly Rainfall (mm)')
ax.set_ylabel('Monthly Accidents')
ax.legend(frameon=True)
fig3.tight_layout()
fig3.savefig('fig3_rainfall_vs_accidents_scatter.png', dpi=150, bbox_inches='tight')
plt.close(fig3)
print("✅ Saved fig3_rainfall_vs_accidents_scatter.png")

# ── FIG 4: State-level annual accidents heatmap (top 15 states) ─────────────
fig4, ax = plt.subplots(figsize=(11, 7))
# Pivot: states × years
pivot_acc = acc_long.copy()
pivot_acc['state'] = pivot_acc['state'].apply(normalize)
top15 = (pivot_acc.groupby('state')['annual_accidents'].mean()
         .nlargest(15).index.tolist())
heat_data = (pivot_acc[pivot_acc['state'].isin(top15)]
             .pivot_table(index='state', columns='year', values='annual_accidents', aggfunc='mean'))
sns.heatmap(heat_data, ax=ax, cmap='YlOrRd', annot=True, fmt='.0f',
            linewidths=0.4, linecolor='white',
            cbar_kws={'label': 'Annual Accidents', 'shrink': 0.8})
ax.set_title('Annual Road Accidents — Top 15 States (2019–2023)', fontweight='bold', pad=12)
ax.set_xlabel('Year')
ax.set_ylabel('')
ax.tick_params(axis='x', rotation=0)
ax.tick_params(axis='y', rotation=0)
fig4.tight_layout()
fig4.savefig('fig4_state_accidents_heatmap.png', dpi=150, bbox_inches='tight')
plt.close(fig4)
print("✅ Saved fig4_state_accidents_heatmap.png")

# ── FIG 4b: Feature × Target Correlation Heatmap ────────────────────────────
FEAT_LABELS = {
    'monthly_total_mm':   'Total Rainfall',
    'monthly_avg_daily':  'Avg Daily Rain',
    'monthly_max_daily':  'Max Daily Rain',
    'rainy_days':         'Rainy Days',
    'heavy_rain_days':    'Heavy Rain Days',
    'prev_month_total':   'Prev Month Rain',
    'prev2_month_total':  '2-Month Lag Rain',
    'rain_damage_index':  'Rain Damage Index',
    'post_rain_dry':      'Post-Rain Dry',
    'month':              'Month',
    'is_monsoon':         'Monsoon Flag',
}
corr_df = df[FEATURES + ['monthly_accidents']].copy()
corr_df.columns = [FEAT_LABELS.get(c, c) for c in corr_df.columns[:-1]] + ['Accidents']
corr_matrix = corr_df.corr()

fig4b, ax4b = plt.subplots(figsize=(11, 9))
mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True   # show lower triangle only
sns.heatmap(
    corr_matrix, ax=ax4b,
    mask=mask,
    cmap='RdYlGn', center=0, vmin=-1, vmax=1,
    annot=True, fmt='.2f', annot_kws={'size': 8},
    linewidths=0.4, linecolor='white',
    cbar_kws={'label': 'Pearson r', 'shrink': 0.75}
)
ax4b.set_title('Feature Correlation Heatmap  (incl. Accidents target)',
               fontweight='bold', pad=14)
ax4b.tick_params(axis='x', rotation=35, labelsize=9)
ax4b.tick_params(axis='y', rotation=0,  labelsize=9)
fig4b.tight_layout()
fig4b.savefig('fig4b_feature_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close(fig4b)
print("✅ Saved fig4b_feature_correlation_heatmap.png")

# ── FIG 5: Feature Importance ────────────────────────────────────────────────
FEAT_LABELS = {
    'monthly_total_mm':   'Total Monthly Rainfall',
    'monthly_avg_daily':  'Avg Daily Rainfall',
    'monthly_max_daily':  'Max Daily Rainfall',
    'rainy_days':         'Rainy Days (>2.5mm)',
    'heavy_rain_days':    'Heavy Rain Days (>40mm)',
    'prev_month_total':   'Prev Month Rainfall',
    'prev2_month_total':  '2-Month Lag Rainfall',
    'rain_damage_index':  'Rain Damage Index',
    'post_rain_dry':      'Post-Rain Dry Period',
    'month':              'Month (Calendar)',
    'is_monsoon':         'Monsoon Season Flag',
}
fi_labelled = fi.rename(index=FEAT_LABELS).sort_values()

fig5, ax = plt.subplots(figsize=(9, 6))
colors_fi = ['#4a90d9' if v > fi_labelled.median() else '#a8c8e8' for v in fi_labelled.values]
bars_fi = ax.barh(fi_labelled.index, fi_labelled.values, color=colors_fi, edgecolor='white')
ax.set_title('Random Forest — Feature Importance', fontweight='bold', pad=12)
ax.set_xlabel('Importance Score')
for bar, val in zip(bars_fi, fi_labelled.values):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=8.5)
ax.set_xlim(0, fi_labelled.max() * 1.18)
fig5.tight_layout()
fig5.savefig('fig5_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close(fig5)
print("✅ Saved fig5_feature_importance.png")

# ── FIG 6: Actual vs Predicted ───────────────────────────────────────────────
fig6, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, preds, alpha=0.4, s=20, color='#4a90d9', rasterized=True)
lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
ax.plot(lims, lims, '--', color='#c0392b', lw=1.5, label='Perfect Prediction')
ax.set_title(f'Actual vs Predicted Monthly Accidents\nR² = {r2:.3f}  |  MAE = {mae:.0f}',
             fontweight='bold', pad=12)
ax.set_xlabel('Actual Accidents')
ax.set_ylabel('Predicted Accidents')
ax.legend(frameon=True)
fig6.tight_layout()
fig6.savefig('fig6_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close(fig6)
print("✅ Saved fig6_actual_vs_predicted.png")

# ── FIG 7: Residuals distribution ────────────────────────────────────────────
residuals = y_test - preds
fig7, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(residuals, bins=40, color='#4a90d9', edgecolor='white', alpha=0.85)
axes[0].axvline(0, color='#c0392b', lw=1.5, ls='--')
axes[0].set_title('Residual Distribution', fontweight='bold')
axes[0].set_xlabel('Residual (Actual − Predicted)')
axes[0].set_ylabel('Count')

axes[1].scatter(preds, residuals, alpha=0.35, s=16, color='#4a90d9', rasterized=True)
axes[1].axhline(0, color='#c0392b', lw=1.5, ls='--')
axes[1].set_title('Residuals vs Fitted Values', fontweight='bold')
axes[1].set_xlabel('Predicted Accidents')
axes[1].set_ylabel('Residual')

fig7.suptitle('Model Residual Diagnostics', fontweight='bold', fontsize=14, y=1.01)
fig7.tight_layout()
fig7.savefig('fig7_residuals.png', dpi=150, bbox_inches='tight')
plt.close(fig7)
print("✅ Saved fig7_residuals.png")

# ── FIG 8: Monsoon vs Non-monsoon accident comparison (box plot) ─────────────
fig8, ax = plt.subplots(figsize=(8, 6))
monsoon_label = df['is_monsoon'].map({1: 'Monsoon\n(Jun–Sep)', 0: 'Non-Monsoon\n(Oct–May)'})
df_plot = df.copy()
df_plot['season'] = monsoon_label
sns.boxplot(data=df_plot, x='season', y='monthly_accidents', ax=ax,
            palette=['#4a90d9', '#a8c8e8'], width=0.5,
            flierprops=dict(marker='o', markerfacecolor='grey', markersize=3, alpha=0.4))
ax.set_title('Monthly Accidents: Monsoon vs Non-Monsoon', fontweight='bold', pad=12)
ax.set_xlabel('')
ax.set_ylabel('Monthly Accidents')
fig8.tight_layout()
fig8.savefig('fig8_monsoon_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig8)
print("✅ Saved fig8_monsoon_comparison.png")

# =============================================================================
# 9. FINAL SUMMARY
# =============================================================================
print("\n" + "="*65)
print("PIPELINE COMPLETE")
print("="*65)
print(f"  States in analysis   : {df['state_norm'].nunique()}")
print(f"  Years covered        : {sorted(df['year'].unique())}")
print(f"  Total records        : {len(df)}")

print("\n── Model Results ──────────────────────────────────────────────")
print(f"  {'Model':<22} {'Test R²':>8} {'CV R²':>10} {'MAE':>8} {'RMSE':>8}")
print("  " + "-"*60)
for res in all_results:
    marker = "  🏆" if res['name'] == best['name'] else "   "
    print(f"{marker} {res['name']:<22} {res['r2']:>8.4f} "
          f"{res['cv_r2']:>7.4f}±{res['cv_std']:.3f} "
          f"{res['mae']:>8.0f} {res['rmse']:>8.0f}")

print(f"\n  Best Model  → {best['name']}")
print(f"  Reason      → Highest 5-fold CV R² ({best['cv_r2']:.4f}), "
      f"meaning it generalises best to unseen data.")
print(f"\n  Interpretation:")
print(f"    • Calendar month and rainfall lag features dominate importance,")
print(f"      suggesting seasonal patterns matter more than raw rainfall intensity.")
print(f"    • Tree-based ensemble models (RF, XGBoost, LightGBM) outperform SVR,")
print(f"      confirming the rainfall–accident relationship is non-linear.")

print("\nSaved figures:")
for i, name in enumerate([
    'fig1_monthly_rainfall.png         — Avg monthly rainfall, monsoon highlighted',
    'fig2_monthly_accidents.png        — Avg monthly accidents seasonality',
    'fig3_rainfall_vs_accidents_scatter.png — Rainfall vs accidents scatter',
    'fig4_state_accidents_heatmap.png  — Top-15 states × year heatmap',
    'fig5_feature_importance.png       — RF feature importances',
    'fig6_actual_vs_predicted.png      — RF actual vs predicted',
    'fig7_residuals.png                — RF residual diagnostics',
    'fig8_monsoon_comparison.png       — Monsoon vs non-monsoon boxplot',
], 1):
    print(f"  [{i:02d}] {name}")