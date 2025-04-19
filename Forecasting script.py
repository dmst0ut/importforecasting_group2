import matplotlib
matplotlib.use('Agg')   # non-interactive backend

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error

# 1. Load data
df = pd.read_csv(
    'C:/Users/Peizeng Lin/Desktop/economic_data.csv',
    parse_dates=['Unnamed: 0']
)
df.rename(columns={'Unnamed: 0': 'DATE'}, inplace=True)
df.set_index('DATE', inplace=True)
df.index = pd.DatetimeIndex(df.index).to_period('Q')

# 2. Compute GDP % growth
df['gdp_growth'] = df['gdp'].pct_change() * 100

# 3. Auto-detect Fed funds rate column
fed_cols = [c for c in df.columns if 'fed' in c.lower()]
if not fed_cols:
    raise KeyError("Fed funds rate column not found (looking for 'fed')")
df['interest_rate'] = df[fed_cols[0]]

# 4. Auto-detect Customs duties column
duty_cols = [c for c in df.columns if 'custom' in c.lower() or 'duti' in c.lower()]
if not duty_cols:
    raise KeyError("Customs duties column not found (looking for 'custom' or 'duti')")
df['customs_duties'] = df[duty_cols[0]]

# 5. Subset & drop NAs
df_model = df[['imports','gdp_growth','interest_rate','customs_duties']].dropna()

# 6. Plot raw series → raw_series.png
plt.figure(figsize=(10,6))
for col in df_model.columns:
    plt.plot(df_model.index.to_timestamp(), df_model[col], label=col)
plt.title('Raw Series: Imports, GDP % Growth, Fed Funds, Customs Duties')
plt.legend()
plt.tight_layout()
plt.savefig('raw_series.png')
plt.close()
print("Saved raw series chart → raw_series.png")

# 7. ADF tests
print("\n=== ADF Tests ===")
for col in df_model.columns:
    stat, p, *_ = adfuller(df_model[col])
    print(f"{col}: ADF={stat:.3f}, p-value={p:.3f}")

# 8. Difference for stationarity
df_diff = df_model.diff().dropna()

# 9. Train/test split (last 6 quarters hold‑out)
h = 6
train_df = df_diff.iloc[:-h]
test_df  = df_diff.iloc[-h:]

# 10. VAR lag‐order selection (maxlags=4)
var = VAR(train_df)
sel = var.select_order(maxlags=4)
lag = sel.aic
print(f"\nSelected lag by AIC: {lag}")

# 11. Fit VAR
model = var.fit(lag)

# 12. Granger causality
print("\n=== Granger Causality ===")
for exog in ['gdp_growth','interest_rate','customs_duties']:
    result = model.test_causality('imports', exog, kind='wald')
    print(f"{exog} → imports: p-value = {result.pvalue:.4f}")

# 13. Forecast next 6 quarters with 95% CI
fc_vals, lower, upper = model.forecast_interval(
    train_df.values[-lag:], steps=h
)
idx = df_diff.columns.get_loc('imports')
last_imp = df_model['imports'].iloc[-h-1]
imp_fc   = fc_vals[:, idx].cumsum() + last_imp
imp_low  = lower[:, idx].cumsum() + last_imp
imp_high = upper[:, idx].cumsum() + last_imp
dates_fc = test_df.index.to_timestamp()

# 14. Plot forecast → imports_forecast_ci.png
plt.figure(figsize=(8,5))
plt.plot(df_model.index.to_timestamp(), df_model['imports'], label='Actual')
plt.plot(dates_fc, imp_fc, '--o', label='Forecast')
plt.fill_between(dates_fc, imp_low, imp_high, alpha=0.3, label='95% CI')
plt.title('Imports: Actual vs VAR Forecast (95% CI)')
plt.legend()
plt.tight_layout()
plt.savefig('imports_forecast_ci.png')
plt.close()
print("Saved forecast chart → imports_forecast_ci.png")
