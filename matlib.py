import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('assets/fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89.csv')

print(df.columns) 

df['Date'] = pd.to_datetime(df['Date'])

df = df[df['Element'].isin(['TMAX', 'TMIN'])].copy()

df['temperature'] = df['Data_Value'] / 10.0

df['year'] = df['Date'].dt.year
df['month_day'] = df['Date'].dt.strftime('%m-%d')

df = df[df['month_day'] != '02-29']

df_2005_2014 = df[(df['year'] >= 2005) & (df['year'] <= 2014)]
df_2015 = df[df['year'] == 2015]

record_high = df_2005_2014[df_2005_2014['Element'] == 'TMAX'].groupby('month_day')['temperature'].max()
record_low  = df_2005_2014[df_2005_2014['Element'] == 'TMIN'].groupby('month_day')['temperature'].min()


record_high_2015 = df_2015[df_2015['Element'] == 'TMAX'].groupby('month_day')['temperature'].max()
record_low_2015  = df_2015[df_2015['Element'] == 'TMIN'].groupby('month_day')['temperature'].min()


dates = pd.to_datetime('2005-' + record_high.index)
day_of_year = dates.dayofyear
df_record = pd.DataFrame({
    'day_of_year': day_of_year,
    'record_high': record_high.values,
    'record_low': record_low.values
}).sort_values('day_of_year')

dates_2015 = pd.to_datetime('2015-' + record_high_2015.index)
day_of_year_2015 = dates_2015.dayofyear
df_record_2015 = pd.DataFrame({
    'day_of_year': day_of_year_2015,
    'record_high_2015': record_high_2015.values,
    'record_low_2015': record_low_2015.values
}).sort_values('day_of_year')


plt.figure(figsize=(12, 6))
plt.plot(df_record['day_of_year'], df_record['record_high'], 
         color='red', linestyle='-', linewidth=2, label='Record High (2005-2014)')
plt.plot(df_record['day_of_year'], df_record['record_low'], 
         color='blue', linestyle='-', linewidth=2, label='Record Low (2005-2014)')
plt.fill_between(df_record['day_of_year'], df_record['record_low'], df_record['record_high'], 
                 color='gray', alpha=0.3)

mask_high = df_record_2015['record_high_2015'] > np.interp(df_record_2015['day_of_year'], 
                                                             df_record['day_of_year'], 
                                                             df_record['record_high'])
mask_low = df_record_2015['record_low_2015'] < np.interp(df_record_2015['day_of_year'], 
                                                           df_record['day_of_year'], 
                                                           df_record['record_low'])

plt.scatter(df_record_2015.loc[mask_high, 'day_of_year'], 
            df_record_2015.loc[mask_high, 'record_high_2015'],
            color='darkred', s=50, marker='o', label='New Record High (2015)')

plt.scatter(df_record_2015.loc[mask_low, 'day_of_year'], 
            df_record_2015.loc[mask_low, 'record_low_2015'],
            color='darkblue', s=50, marker='o', label='New Record Low (2015)')

plt.title('Daily Record High and Low Temperatures (2005-2014) with 2015 Extremes')
plt.xlabel('Day of Year')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.tight_layout()
plt.show()
