## ASSIGNMENT 1

**PREPARE DATA**

import pandas as pd
import numpy as np
from math import sqrt

df = pd.read_csv('https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/current.csv?sc_lang=en&hash=80445D12401C59CF716410F3F7863B64')

df_cleaned = df.drop(index=0)
df_cleaned.reset_index(drop=True, inplace=True)
df_cleaned['sasdate'] = pd.to_datetime(df_cleaned['sasdate'], format='%m/%d/%Y')
df_cleaned

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

series_to_plot = ['INDPRO', 'CPIAUCSL', 'TB3MS']
series_names = ['Industrial Production',
                'Inflation (CPI)',
                '3-month Treasury Bill rate']

fig, axs = plt.subplots(len(series_to_plot), 1, figsize=(8, 15))

for ax, series_name, plot_title in zip(axs, series_to_plot, series_names):
    if series_name in df_cleaned.columns:
        dates = pd.to_datetime(df_cleaned['sasdate'], format='%m/%d/%Y')
        ax.plot(dates, df_cleaned[series_name], label=plot_title)
        ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_title(plot_title)
        ax.set_xlabel('Year')
        ax.set_ylabel('Index')
        ax.legend(loc='upper left')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax.set_visible(False)

plt.tight_layout()
plt.show()

"""The variables are non-stationary, as values vary with time. We also condect Autocorrelation test as below:"""

# Check for autocorrelation with Autocorrelation Function (ACF)
from statsmodels.graphics.tsaplots import plot_acf

series = df_cleaned['INDPRO']
series2 = df_cleaned['CPIAUCSL']
series3 = df_cleaned['TB3MS']

# Create a figure and subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 12))
plot_acf(series, lags=25, ax=axes[0], title='Autocorrelation of INDPRO')
plot_acf(series2, lags=25, ax=axes[1], title='Autocorrelation of CPIAUCSL')
plot_acf(series3, lags=25, ax=axes[2], title='Autocorrelation of TB3MS')

for ax in axes:
    ax.set_xlabel('Lag')
    ax.set_ylabel('Correlation')

plt.tight_layout()
plt.show()

"""Since there are autocorrelation in all 3 variables, the transformation with differencing is necessary. From csv file of the Dataset, the 2nd row indicating the transformation code for us to apply to reach stationary."""

# First selects the first row (0) and all columns starting from the second one (1:) of the DataFrame df. This row contains the transformation codes.

transformation_codes = df.iloc[0, 1:].to_frame().reset_index()
transformation_codes.columns = ['Series', 'Transformation_Code']

def apply_transformation(series, code):
    if code == 1:
        return series
    elif code == 2:
        return series.diff()
    elif code == 3:
        return series.diff().diff()
    elif code == 4:
        return np.log(series)
    elif code == 5:
        return np.log(series).diff()
    elif code == 6:
        return np.log(series).diff().diff()
    elif code == 7:
        return series.pct_change()
    else:
        raise ValueError("Invalid transformation code")

for series_name, code in transformation_codes.values:
    df_cleaned[series_name] = apply_transformation(df_cleaned[series_name].astype(float), float(code))

df_cleaned = df_cleaned[2:]
df_cleaned.reset_index(drop=True, inplace=True)
df_cleaned.head()

series_to_plot = ['INDPRO', 'CPIAUCSL', 'TB3MS']
series_names = ['Industrial Production',
                'Inflation (CPI)',
                '3-month Treasury Bill rate']

fig, axs = plt.subplots(len(series_to_plot), 1, figsize=(8, 15))

for ax, series_name, plot_title in zip(axs, series_to_plot, series_names):
    if series_name in df_cleaned.columns:
        dates = pd.to_datetime(df_cleaned['sasdate'], format='%m/%d/%Y')
        ax.plot(dates, df_cleaned[series_name], label=plot_title)
        ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_title(plot_title)
        ax.set_xlabel('Year')
        ax.set_ylabel('Transformed Value')
        ax.legend(loc='upper left')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax.set_visible(False)

plt.tight_layout()
plt.show()

"""The data is now presented in 3 graphs, now we turns to forecasting

We conduct a quick check of Autocorrelation again, this time with the transformed values. Also we find the proper lags good for predictive model.
"""

# Check which lag variable good for predictive model
# plot with Autocorrelation Function
from statsmodels.graphics.tsaplots import plot_pacf

series = df_cleaned['INDPRO']
series2 = df_cleaned['CPIAUCSL']
series3 = df_cleaned['TB3MS']

# Create a figure and subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 12))
plot_acf(series, lags=25, ax=axes[0], title='Autocorrelation of INDPRO after transformation')
plot_acf(series2, lags=25, ax=axes[1], title='Autocorrelation of CPIAUCSL afer transformation')
plot_acf(series3, lags=25, ax=axes[2], title='Autocorrelation of TB3MS after transformation')

for ax in axes:
    ax.set_xlabel('Lag')
    ax.set_ylabel('Correlation')

plt.tight_layout()
plt.show()

"""From the result of Autocorrelation testing, we may conclude:
* High autocorrelation at lag 1 -> a strong relationship with the immediate past value.
* Most of the other lags within confidence interval - not significantly different from zero - past values beyond the immediate previous one do not strongly influence the current value.
* No seasonality in the data.
* The series likely follows a short-memory process with the correlation within CI for all three variables at lag -> AR(4) model might be appropriate for forecasting.

---

FORECAST IN TIME SERIES
predicting the log-difference of industrial production
*   INDPRO: target variable
*   CPIAUSL and TB3MS is predictors.

From Linear model: Y = XB + u.
Create matrix X, vector Y.
Save last row of X for forecasting (convert to numpy)
"""

Yraw = df_cleaned['INDPRO']
Xraw = df_cleaned[['CPIAUCSL', 'TB3MS']]

num_lags  = 4 #p
num_leads = 1 #h-step

X = pd.DataFrame()

col = 'INDPRO'
for lag in range(0,num_lags+1):
        X[f'{col}_lag{lag}'] = Yraw.shift(lag)

for col in Xraw.columns:
    for lag in range(0,num_lags+1):
        X[f'{col}_lag{lag}'] = Xraw[col].shift(lag)
X.insert(0, 'Ones', np.ones(len(X)))

X.head()

y = Yraw.shift(-num_leads)
y

X_T = X.iloc[-1:].values

y = y.iloc[num_lags:-num_leads].values
X = X.iloc[num_lags:-num_leads].values

X_T

"""Estimate parameters and obtain forecast: Finding B with OLS"""

from numpy.linalg import solve

beta_ols = solve(X.T @ X, X.T @ y)

forecast = X_T@beta_ols*100
forecast

"""Real-time evaluation: set T = 12/1999 > Estimate model till T > Product Yhat of T+1, T+2,...T+H > Then caculate forecasting errors (MSFE) and Rooted Mean Errors (RMSFE) for each steps h = 1,4,8

Define the calculation of forecast as follow
"""

def calculate_forecast(df_cleaned, p = 4, H = [1,4,8], end_date = '12/1/1999',target = 'INDPRO', xvars = ['CPIAUCSL', 'TB3MS']):

    ## Subset df_cleaned to use only data up to end_date
    rt_df = df_cleaned[df_cleaned['sasdate'] <= pd.Timestamp(end_date)]
    ## Get the actual values of target at different steps ahead
    Y_actual = []
    for h in H:
        os = pd.Timestamp(end_date) + pd.DateOffset(months=h)
        Y_actual.append(df_cleaned[df_cleaned['sasdate'] == os][target]*100)
        ## Now Y contains the true values at T+H (multiplying * 100)

    Yraw = rt_df[target]
    Xraw = rt_df[xvars]

    X = pd.DataFrame()
    ## Add the lagged values of Y
    for lag in range(0,p):
        # Shift each column in the DataFrame and name it with a lag suffix
        X[f'{target}_lag{lag}'] = Yraw.shift(lag)

    for col in Xraw.columns:
        for lag in range(0,p):
            X[f'{col}_lag{lag}'] = Xraw[col].shift(lag)

    ## Add a column on ones (for the intercept)
    X.insert(0, 'Ones', np.ones(len(X)))

    ## Save last row of X (converted to numpy)
    X_T = X.iloc[-1:].values

    ## While the X will be the same, Y needs to be leaded differently
    Yhat = []
    for h in H:
        y_h = Yraw.shift(-h)
        ## Subset getting only rows of X and y from p+1 to h-1
        y = y_h.iloc[p:-h].values
        X_ = X.iloc[p:-h].values
        # Solving for the OLS estimator beta: (X'X)^{-1} X'Y
        beta_ols = solve(X_.T @ X_, X_.T @ y)
        ## Produce the One step ahead forecast
        ## % change month-to-month INDPRO
        Yhat.append(X_T@beta_ols*100)

    ## Now calculate the forecasting error and return

    return np.array(Y_actual) - np.array(Yhat)

t0 = pd.Timestamp('12/1/1999')
e = []
T = []
for j in range(0, 10):
    t0 = t0 + pd.DateOffset(months=1)
    print(f'Using data up to {t0}')
    ehat = calculate_forecast(df_cleaned, p = 4, H = [1,4,8], end_date = t0)
    e.append(ehat.flatten())
    T.append(t0) #Add current end date t0 to list T - store all end dates

## Create a pandas DataFrame from the list
edf_p4 = pd.DataFrame((e), columns=["h=1", "h=4", "h=8"])

# Compute MSFE and RMSE
msfe_p4 = edf_p4.apply(np.square).mean()
rmse_p4 = np.sqrt(msfe_p4)

results_df_p4 = pd.DataFrame({"MSFE": msfe_p4, "RMSE": rmse_p4})
print(results_df_p4)

"""The results give the intuition that: Longer-term forecasts tend to have higher errors, which is the typical case.

Now we may test for different models, with different lags. For example, given lag = 1
"""

def calculate_forecast_p1(df_cleaned, p=1, H=[1,4,8], end_date='12/1/1999', target='INDPRO', xvars=['CPIAUCSL', 'TB3MS']):
    rt_df = df_cleaned[df_cleaned['sasdate'] <= pd.Timestamp(end_date)]

    Y_actual = []
    for h in H:
        os = pd.Timestamp(end_date) + pd.DateOffset(months=h)
        Y_actual.append(df_cleaned[df_cleaned['sasdate'] == os][target] * 100)  # Convert to percentage

    Yraw = rt_df[target]
    Xraw = rt_df[xvars]

    X = pd.DataFrame()
    X[f'{target}_lag1'] = Yraw.shift(1)

    for col in Xraw.columns:
        X[f'{col}_lag1'] = Xraw[col].shift(1)

    X.insert(0, 'Ones', np.ones(len(X)))

    X_T = X.iloc[-1:].values

    Yhat = []
    for h in H:
        y_h = Yraw.shift(-h)
        y = y_h.iloc[p:-h].values
        X_ = X.iloc[p:-h].values

        beta_ols = solve(X_.T @ X_, X_.T @ y)

        Yhat.append(X_T @ beta_ols * 100)

    return np.array(Y_actual) - np.array(Yhat)

t0 = pd.Timestamp('12/1/1999')
e = []
T = []
for j in range(0, 10):
    t0 += pd.DateOffset(months=1)
    print(f'Using data up to {t0}')
    ehat = calculate_forecast_p1(df_cleaned, p=1, H=[1,4,8], end_date=t0)
    e.append(ehat.flatten())
    T.append(t0)

edf_p1 = pd.DataFrame(e, columns=["h=1", "h=4", "h=8"])

msfe_p1 = edf_p1.apply(np.square).mean()
rmse_p1 = np.sqrt(msfe_p1)

results_df_p1 = pd.DataFrame({"MSFE": msfe_p1, "RMSE": rmse_p1})
print(results_df_p1)

"""With lag = 1, RMSE is lower in case h=1 (0.32 vs 0.34) and higher in case h=4 and h=8. The model predicts better in 1-step-ahead forecast only, compared to model with lag = 4.

To show this graphically:
"""

# Extract RMSE values
rmse_p4 = results_df_p4['RMSE'].values
rmse_p1 = results_df_p1['RMSE'].values

# Forecast steps (x-axis)
steps = [1, 4, 8]

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(steps, rmse_p4, marker='o', label='p=4')
plt.plot(steps, rmse_p1, marker='o', label='p=1')

# Set plot properties
plt.xlabel('Forecast Steps (h)')
plt.ylabel('RMSE')
plt.title('RMSE Comparison for Different Lags (p)')
plt.xticks(steps)
plt.ylim(0, 1)
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# Calculate Residual Standard Deviation (RSD) to evaluate model
def calculate_rsd(rmse, edf):
    a = rmse - edf.apply(np.square)
    b = a.mean()
    return np.sqrt(b)

rmse_value = results_df_p4['RMSE'].values
edf = pd.DataFrame(e, columns=["h=1", "h=4", "h=8"])

rsd_value = calculate_rsd(rmse_value, edf)
print(edf)
print("Residual Standard Deviation (RSD):", rsd_value)

def calculate_forecast_p4_test(df_cleaned, p=4, H=[1, 4, 8], start_date='1/1/2001', end_date='12/1/2024', target='INDPRO', xvars=['CPIAUCSL', 'TB3MS']):
    # Subset the data for the test period
    test_df = df_cleaned[(df_cleaned['sasdate'] >= pd.Timestamp(start_date)) & (df_cleaned['sasdate'] <= pd.Timestamp(end_date))]

    e = []
    T = []

    # Iterate through the test set
    for index in test_df.index[:-max(H)]:  # Exclude last H periods for forecasting
        current_date = test_df.loc[index, 'sasdate']
        # print(f'Using data up to {current_date}')

        # Get actual values for the forecast horizons
        Y_actual = []
        for h in H:
            forecast_date = pd.Timestamp(current_date) + pd.DateOffset(months=h)
            actual_value = df_cleaned[df_cleaned['sasdate'] == forecast_date][target].values[0] * 100

        # Call calculate_forecast using data up to current_date for estimation (modified)
        ehat = calculate_forecast(df_cleaned, p=p, H=H, end_date=current_date, target=target, xvars=xvars)
        e.append(ehat.flatten())  # Store forecast errors
        T.append(current_date)  # Store current date

    # Create DataFrame from errors and calculate MSFE, RMSE
    edf_p4 = pd.DataFrame(e, columns=["h=1", "h=4", "h=8"])
    msfe_p4 = edf_p4.apply(np.square).mean()
    rmse_p4 = np.sqrt(msfe_p4)
    results_df_p4 = pd.DataFrame({"MSFE": msfe_p4, "RMSE": rmse_p4})

    return results_df_p4

results_test = calculate_forecast_p4_test(df_cleaned)
print(results_test)

def calculate_forecast_p1_test(df_cleaned, p=1, H=[1, 4, 8], start_date='1/1/2001', end_date='12/1/2024', target='INDPRO', xvars=['CPIAUCSL', 'TB3MS']):
    # Subset the data for the test period
    test_df = df_cleaned[(df_cleaned['sasdate'] >= pd.Timestamp(start_date)) & (df_cleaned['sasdate'] <= pd.Timestamp(end_date))]

    e = []  # List to store forecast errors
    T = []  # List to store evaluation dates

    # Iterate through the test set
    for index in test_df.index[:-max(H)]:  # Exclude last H periods for forecasting
        current_date = test_df.loc[index, 'sasdate']

        Y_actual = []
        for h in H:
            forecast_date = pd.Timestamp(current_date) + pd.DateOffset(months=h)
            actual_value = df_cleaned[df_cleaned['sasdate'] == forecast_date][target].values[0] * 100
            Y_actual.append(actual_value)

        # Call calculate_forecast_p1 using data up to current_date for estimation
        ehat = calculate_forecast_p1(df_cleaned, p=p, H=H, end_date=current_date, target=target, xvars=xvars)
        e.append(ehat.flatten())  # Store forecast errors
        T.append(current_date)  # Store current date

    # Create DataFrame from errors and calculate MSFE, RMSE
    edf_p1 = pd.DataFrame(e, columns=["h=1", "h=4", "h=8"])
    msfe_p1 = edf_p1.apply(np.square).mean()
    rmse_p1 = np.sqrt(msfe_p1)
    results_df_p1 = pd.DataFrame({"MSFE": msfe_p1, "RMSE": rmse_p1})

    return results_df_p1
    return results

# Call the function to apply the model to the test set
results_test = calculate_forecast_p1_test(df_cleaned)
print(results_test)

"""When applying the defined model to forecast from 1/1/2001 to 1/1/2024, the model seems to be overfitted as RMSE increase double. The reason could be using too board timeframe."""

## Other problems to be solved:
# Fitting the model
# Plot predicted vs actual data, aiming for the plot line closest to diagonal
