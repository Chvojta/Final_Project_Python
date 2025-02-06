import pandas as pd

input_file_path = "Inputs.xlsx"
benchmark_file_path ="Benchmarks.xlsx"

# Load the Excel file (all sheets)
df_dict = pd.read_excel(benchmark_file_path, sheet_name=None)

# Initialize assumptions storage
assumptions = {}

# Process each sheet in the file
for sheet_name, df in df_dict.items():
    if df.empty:
        continue  # Skip empty sheets

    df = df.dropna(how='all')  # Drop fully empty rows
    row_headers = df.iloc[:, 0]  # First column as row headers (assumed unique)
    col_headers = df.columns[1:]  # All other columns

    # Iterate through each row/column pair to store the actual value from the sheet
    for row_idx, row_header in enumerate(row_headers):
        for col_idx, col_header in enumerate(col_headers):
            value = df.iloc[row_idx, col_idx + 1]  # Get the actual value from the DataFrame
            assumptions[(sheet_name, row_header, col_header)] = value  # Store actual value

# Print the first 3-4 assumptions per sheet
for sheet_name in df_dict.keys():
    print(f"\nSheet: {sheet_name}")
    count = 0
    for key in assumptions.keys():
        if key[0] == sheet_name:
            print(f"  - Row: {key[1]}, Column: {key[2]}, Assumption: {assumptions[key]}")
            count += 1
        if count >= 4:  # Print only first 3-4 per sheet
            break


