import pandas as pd
import os
import calendar
import math

def clean_and_update_chiller_data(input_dir='csvs'):
    output_file = os.path.join(input_dir, 'chillers.csv')  # detailed output
    sum_output_file = os.path.join(input_dir, 'sum_chillers.csv')  # summary output

    # Find CDD and chillers files
    CDD_file = None
    chiller_file = None
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.csv'):
            if 'CDD' in file_name:
                CDD_file = os.path.join(input_dir, file_name)
            elif 'Chiller' in file_name or 'EnPI' in file_name:
                chiller_file = os.path.join(input_dir, file_name)

    if not CDD_file or not chiller_file:
        raise FileNotFoundError("Could not find both CDD and chillers CSV files in the 'csvs' folder.")


    # Read and clean CDD data (skip metadata rows)
    CDD = pd.read_csv(CDD_file, skiprows=6)
    if 'Date' not in CDD.columns:
        raise ValueError(f"CDD file {CDD_file} does not have a 'Date' column. Columns: {CDD.columns}")
    CDD['Date'] = pd.to_datetime(CDD['Date'], errors='coerce').dt.date
    CDD = CDD.dropna(subset=['Date'])

    # Read and clean chillers data
    chillers = pd.read_csv(chiller_file)
    chillers.rename(columns={col: col.replace(" (kg)", "") for col in chillers.columns}, inplace=True)
    chillers.fillna(0, inplace=True)
    chillers['Timestamp'] = pd.to_datetime(chillers['Timestamp'])
    chillers['Date'] = chillers['Timestamp'].dt.date
    chillers['Time'] = chillers['Timestamp'].dt.time

    # Merge cleaned data (inner join on Date)
    combined = pd.merge(chillers, CDD, on='Date', how='inner')

    # Add Year, Month, Week, Day columns for dashboard compatibility
    combined['Year'] = pd.to_datetime(combined['Date']).dt.year.astype(str)
    combined['Month'] = pd.to_datetime(combined['Date']).dt.strftime('%b')
    combined['Week'] = pd.to_datetime(combined['Date']).dt.isocalendar().week.astype(int)
    combined['Day'] = pd.to_datetime(combined['Date']).dt.strftime('%a')

    # Output detailed file (like steam.csv)
    detailed_cols = ['Timestamp', 'Date', 'Time', 'Year', 'Month', 'Week', 'Day', 'CDD 15.5', 'Chiller 03 2Q3', 'Chiller 02 2Q9', 'Chiller 01 2Q10', 'Chiller CH-WC07301']
    combined = combined[detailed_cols]


    # Save detailed file (overwrite, like steam.csv)
    combined.to_csv(output_file, index=False)
    print(f"Detailed chiller file saved: {output_file}")

    # Prepare summary (like sum_steam.csv): sum by date, keep CDD for each date
    meter_cols = ['Chiller 03 2Q3', 'Chiller 02 2Q9', 'Chiller 01 2Q10', 'Chiller CH-WC07301']
    summed = combined.groupby('Date', as_index=False)[meter_cols].sum(numeric_only=True)
    # Get CDD value for each date (first value for that date)
    CDD_vals = combined.groupby('Date', as_index=False)['CDD 15.5'].first()
    summed = pd.merge(summed, CDD_vals, on='Date', how='left')
    # Reorder columns to match steam summary style
    summary_cols = ['Date'] + meter_cols + ['CDD 15.5']
    summed = summed[summary_cols]

    if os.path.exists(sum_output_file):
        existing_sum = pd.read_csv(sum_output_file)
        existing_sum['Date'] = pd.to_datetime(existing_sum['Date']).dt.date
        summed['Date'] = pd.to_datetime(summed['Date']).dt.date
        summed = pd.concat([existing_sum, summed], ignore_index=True).drop_duplicates(subset=['Date'], keep='last')

    summed.to_csv(sum_output_file, index=False)
    print(f"Simple format file saved: {sum_output_file}")
    return combined, summed

if __name__ == "__main__":
    clean_and_update_chiller_data()
