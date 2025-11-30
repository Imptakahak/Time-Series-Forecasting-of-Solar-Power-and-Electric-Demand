import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_forecast_vs_actual(results_df, title, days_to_show=10):
    '''
    Plot forecast results and actual values for the specified period.
    results_df: DataFrame with datetime index and columns 'true' and 'pred'
    '''
    # Determine the period to plot
    start_datetime = results_df.index.min()
    end_datetime = start_datetime + pd.Timedelta(days=days_to_show)
    plot_data = results_df.loc[start_datetime:end_datetime]

    if plot_data.empty:
        print(f"Error: No plot data found starting from {start_datetime}.")
        return

    plt.figure(figsize=(15, 7))
    plt.plot(plot_data.index, plot_data['true'], label='Actual', color='blue')
    plt.plot(plot_data.index, plot_data['pred'], label='Forecast', color='red', linestyle='--')
    
    plt.title(title, fontsize=16)
    plt.xlabel('Datetime', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    # X軸のフォーマットを調整
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=max(1, days_to_show // 10)))
    plt.gcf().autofmt_xdate()
    
    plt.show()
