# ==============================
#  TABLE OF CONTENTS (Short)
# ==============================

# 0. Setup (Drive + Imports)
# 1. Load Data
# 2. Row Lookup + Indices
# 3. Cache Previous Rows
# 4. Cancellation Counts
# 5. No-Show / Left
# 6. Provider & Dept Changes
# 7. Duration Stats
# 8. Rescheduled Count
# 9. Time Features
# 10. Encounter Frequency
# 11. First vs Follow-Up
# 12. Last Appointment Status
# 13. Schedule/Arrival Lags
# 14. Time Since Last Visit
# 15. Last Cancel Category
# 16. Last Visit/Encounter Types
# 17. Provider Interpreter Flag
# 18. Save Final Data

# Import necessary libraries
import os
import pandas as pd
import ast  # To safely evaluate the list-like strings into actual lists

# ----------------------------- #
# Step 1: Load the DataFrame (local path)
# ----------------------------- #
_script_dir = os.path.dirname(os.path.abspath(__file__))
csv_load_path = os.path.join(_script_dir, 'df_with_valid_indices_0717.csv')

# Load the CSV into a DataFrame
df = pd.read_csv(csv_load_path)

# Display the first few rows of the DataFrame to confirm it's loaded correctly
print(df.head())

# Retrieve and display column names
column_names = df.columns.tolist()
print(column_names)



print(df.columns)


# ----------------------------- #
# Step 2: Create Row Lookup Dictionary
# ----------------------------- #
# Create a dictionary that maps Row_Index to its corresponding row
row_lookup = {idx: row.to_dict() for idx, row in df.iterrows()}


df['Row_Index'] = df.index



# ----------------------------- #
# Step 3: Convert String to List
# ----------------------------- #
# Function to safely convert string representation of lists to Python lists
def convert_to_list(val):
    if isinstance(val, str):
        try:
            # Safely evaluate the string to a Python list
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []  # Return an empty list if the conversion fails
    return val

# Apply the conversion to the Valid_Previous_Indices column
df['valid_previous_indices'] = df['valid_previous_indices'].apply(convert_to_list)


# ----------------------------- #
# Step 4: Precompute Valid Previous Rows
# ----------------------------- #
# Function to retrieve previous rows based on Valid_Previous_Indices
def get_previous_rows(row, row_lookup):
    previous_indices = row['valid_previous_indices']

    # Retrieve previous rows using the row_lookup dictionary
    previous_rows = [row_lookup.get(i, None) for i in previous_indices]

    # Filter out None values (in case the index is not found)
    return [r for r in previous_rows if r is not None]

# Apply the function to precompute the previous rows
df['Previous_Rows_Cached'] = df.apply(lambda row: get_previous_rows(row, row_lookup), axis=1)

# Display rows with non-empty cached previous rows for verification
non_empty_cached_rows = df[df['Previous_Rows_Cached'].apply(lambda x: len(x) > 0)]
print(non_empty_cached_rows[['Row_Index', 'valid_previous_indices', 'Previous_Rows_Cached']].head())


# ----------------------------- #
# Step 5: Count Cancellations
# ----------------------------- #
# Function to count cancellations in Previous_Rows_Cached
def count_cancellations(row):
    previous_rows = row['Previous_Rows_Cached']

    # Filter for rows where STATUS_CD == 'Canceled'
    cancellation_count = sum(1 for r in previous_rows if r['STATUS_CD'] == 'Canceled')

    return cancellation_count

# Apply the function to count cancellations for each row
df['Cancellation_Count'] = df.apply(count_cancellations, axis=1)

# Display relevant columns for verification
print(df[['Row_Index', 'valid_previous_indices', 'STATUS_CD', 'Cancellation_Count']].head(20))


# ----------------------------- #
# Step 6: Categorize Cancellations by Reason
# ----------------------------- #
# Define the categories and their corresponding reasons
reason_category_mapping = {
    'Patient Choice': [
        'Canceled via automated reminder system', 'Cancelled via Interface', 'Deleted via Interface',
        'Moved', 'Patient', 'Patient - Personal','Personal Reasons', 'Patient - Sought Care Elsewhere', 'Unhappy/Changed Provider','Sought Care Elsewhere'
    ],
    'Clinic Management': [
        'Changed by Radiology', 'Discharged', 'Displaced Appointment', 'Edu/Meeting', 'Error',
        'Institution', 'Order Discontinued', 'Prep/Med/Results Unavailable', 'Schedule Order Error',
        'Scheduled from Wait List','Institution - Appt Made in Error', 'Cancelled via automated reminder system','Level of Care Change','Institution - Condition Warrants Cancellation'
    ],
    'Patient Health': [
        'Clinically Caused', 'Deceased', 'Feeling Better', 'Hospitalized', 'Labs Out of Acceptable Range',
        'Level of Care Change', 'Oncology Treatment Plan Changes','Patient Dismissed From Practice'
    ],
    'Socioeconomic': [
        'Financial', 'Lack of Transportation','Financial Concerns'
    ],
    'Provider Choice': [
        'MD Appointment', 'Provider', 'Provider - Personal', 'Provider - Professional','Provider Departure'
    ]
}
# Flatten the dictionary to map each reason to its corresponding category
reason_to_category = {reason: category for category, reasons in reason_category_mapping.items() for reason in reasons}

# Function to count cancellations by category
def count_cancellations_by_category(row, reason_to_category):
    previous_rows = row['Previous_Rows_Cached']

    # Initialize counters for each category
    category_counts = {category: 0 for category in reason_category_mapping.keys()}

    # Filter for rows where STATUS_CD == 'Canceled' and count reasons by category
    for r in previous_rows:
        if r['STATUS_CD'] == 'Canceled':
            reason = r.get('CNCL_REASON_DESCR')
            if reason in reason_to_category:
                category = reason_to_category[reason]
                category_counts[category] += 1

    return category_counts

# Apply the function to count cancellations by category for each row
category_counts_df = df.apply(lambda row: count_cancellations_by_category(row, reason_to_category), axis=1)

# Convert the result into separate columns for each category
category_counts_df = pd.DataFrame(category_counts_df.tolist(), index=df.index)

# Concatenate the new category count columns with the original dataframe
df = pd.concat([df, category_counts_df], axis=1)

# Display the updated dataframe with category count columns
print(df[['Row_Index', 'valid_previous_indices'] + list(reason_category_mapping.keys())].head(30))

# ----------------------------- #
# Step 7: Count 'No Show' or 'Left Without Seen'
# ----------------------------- #
# Function to count occurrences of 'No Show' or 'Left without seen'
def count_no_show_or_left(row):
    previous_rows = row['Previous_Rows_Cached']

    # Count how many times STATUS_CD is either 'No Show' or 'Left without seen'
    count = sum(1 for r in previous_rows if r['STATUS_CD'] in ['No Show', 'Left without seen'])

    return count

# Apply the function to count for each row
df['No_Show_Left_Count'] = df.apply(count_no_show_or_left, axis=1)

# Display the updated dataframe with the new column
print(df[['Row_Index', 'valid_previous_indices', 'No_Show_Left_Count']].head())

# ----------------------------- #
# Step 8: Track Provider Changes (Optimized)
# ----------------------------- #

def provider_change_stats_optimized(previous_rows, current_provider):
    # Filter ENCOUNTER_PROV_IDs of non-canceled visits
    valid_providers = [
        r['ENCOUNTER_PROV_ID']
        for r in previous_rows
        if r['STATUS_CD'] != 'Canceled' and pd.notna(r['ENCOUNTER_PROV_ID'])
    ]

    # Count provider changes
    provider_change_count = sum(
        valid_providers[i] != valid_providers[i - 1]
        for i in range(1, len(valid_providers))
    )

    # Check if the provider changed in the last valid visit
    provider_changed_last_visit = (
        valid_providers[-1] != current_provider if valid_providers else False
    )

    return provider_change_count, provider_changed_last_visit


# Precompute stats for all rows
provider_stats = [
    provider_change_stats_optimized(row['Previous_Rows_Cached'], row['ENCOUNTER_PROV_ID'])
    for _, row in df.iterrows()
]

# Convert the results into a DataFrame
provider_stats_df = pd.DataFrame(provider_stats, columns=['Provider_Change_Count', 'Provider_Changed_Last_Visit'])

# Add the results back to the original DataFrame
df[['Provider_Change_Count', 'Provider_Changed_Last_Visit']] = provider_stats_df

# Display the updated DataFrame
print(df[['Row_Index', 'valid_previous_indices', 'Provider_Change_Count', 'Provider_Changed_Last_Visit']].head())


# ----------------------------- #
# Step 9: Optimized Department Change Statistics
# ----------------------------- #

# Optimized function to calculate department change statistics
def department_change_stats_optimized(previous_rows, current_department):
    # Filter CLIN_DEPT_ABBREV of non-canceled visits
    valid_departments = [
        r['CLIN_DEPT_ABBREV']
        for r in previous_rows
        if r['STATUS_CD'] != 'Canceled' and pd.notna(r['CLIN_DEPT_ABBREV'])
    ]

    # Count department changes
    department_change_count = sum(
        valid_departments[i] != valid_departments[i - 1]
        for i in range(1, len(valid_departments))
    )

    # Check if the department changed in the last valid visit
    department_changed_last_visit = (
        valid_departments[-1] != current_department if valid_departments else False
    )

    return department_change_count, department_changed_last_visit


# Precompute department stats for all rows
department_stats = [
    department_change_stats_optimized(row['Previous_Rows_Cached'], row['CLIN_DEPT_ABBREV'])
    for _, row in df.iterrows()
]

# Convert the results into a DataFrame
department_stats_df = pd.DataFrame(department_stats, columns=['Department_Change_Count', 'Department_Changed_Last_Visit'])

# Add the results back to the original DataFrame
df[['Department_Change_Count', 'Department_Changed_Last_Visit']] = department_stats_df

# Display the updated DataFrame
print(df[['Row_Index', 'PT_ID', 'valid_previous_indices', 'Department_Change_Count', 'Department_Changed_Last_Visit']].head())


print(df['Previous_Rows_Cached'].head())


import numpy as np
# ----------------------------- #
# Step 10: Valid Duration Statistics (Convert to Hours)
# ----------------------------- #

# Precompute valid durations (in hours) for all rows
def extract_valid_durations_in_hours(previous_rows):
    # Extract STD_DURATION for valid rows and convert from minutes to hours
    return [
        r['STD_DURATION'] / 60  # Convert minutes to hours
        for r in previous_rows
        if r['STATUS_CD'] != 'Canceled' and pd.notna(r['STD_DURATION'])
    ]

# Extract valid durations in hours in batch
df['Valid_Durations'] = df['Previous_Rows_Cached'].map(extract_valid_durations_in_hours)

# Precompute last visit duration and average duration (in hours)
def compute_duration_stats_in_hours(valid_durations):
    if not valid_durations:
        return pd.NA, pd.NA  # Return NaN if no valid durations

    # Calculate the last valid duration and average duration
    last_visit_duration = valid_durations[-1]
    avg_duration = np.mean(valid_durations)

    return last_visit_duration, avg_duration

# Compute duration stats in batch
duration_stats = df['Valid_Durations'].map(compute_duration_stats_in_hours)

# Convert the results into separate columns
df[['Last_Visit_Duration', 'Avg_STD_DURATION']] = pd.DataFrame(duration_stats.tolist(), index=df.index)

# Display the updated DataFrame
print(df[['Row_Index', 'valid_previous_indices', 'Last_Visit_Duration', 'Avg_STD_DURATION']].head())



print(df[['Row_Index', 'valid_previous_indices', 'Last_Visit_Duration', 'Avg_STD_DURATION']].head(20))


# ----------------------------- #
# Step 11: Optimized Rescheduled Appointments Count
# ----------------------------- #

# Define a function to count rescheduled appointments
def count_rescheduled(previous_rows):
    # Count rows where RESCHD_DTTM is not null
    return sum(pd.notna(r['RESCHD_DTTM']) for r in previous_rows)

# Compute rescheduled appointments count for all rows
df['Rescheduled_Appointments_Count'] = df['Previous_Rows_Cached'].map(count_rescheduled)

# Display the updated DataFrame
print(df[['Row_Index', 'valid_previous_indices', 'Rescheduled_Appointments_Count']].head())


# ----------------------------- #
# Step 12: Optimized Time-Related Features
# ----------------------------- #

# Ensure ENCOUNTER_DTTM is in datetime format
df['ENCOUNTER_DTTM'] = pd.to_datetime(df['ENCOUNTER_DTTM'], errors='coerce')

# Extract time-related features using vectorized operations
df['Encounter_Month'] = df['ENCOUNTER_DTTM'].dt.month  # Extract the month
df['Encounter_Weekday'] = df['ENCOUNTER_DTTM'].dt.day_name()  # Extract the weekday
df['Weekend_Indicator'] = (df['ENCOUNTER_DTTM'].dt.weekday >= 5).astype(int)  # Weekend indicator

# Define categories for encounter hour directly with numpy
hour_bins = [0, 12, 18, 24]
hour_labels = ["before 12pm", "between 12-18pm", "after 18pm"]
df['Encounter_Hour'] = pd.cut(df['ENCOUNTER_DTTM'].dt.hour, bins=hour_bins, labels=hour_labels, right=False)

# Display the updated DataFrame
print(df[['Row_Index', 'ENCOUNTER_DTTM', 'Encounter_Month', 'Encounter_Weekday', 'Weekend_Indicator', 'Encounter_Hour']].head())


# ----------------------------- #
# Step 13: Optimized Average Encounter Frequency
# ----------------------------- #

# Precompute valid encounter times in batch
def extract_valid_encounters(previous_rows):
    # Extract ENCOUNTER_DTTM for valid rows and convert to datetime
    valid_encounters = [
        r['ENCOUNTER_DTTM']
        for r in previous_rows
        if r['STATUS_CD'] != 'Canceled' and pd.notna(r['ENCOUNTER_DTTM'])
    ]
    return pd.to_datetime(valid_encounters, errors='coerce')

# Compute valid encounter times for all rows
df['Valid_Encounter_Times'] = df['Previous_Rows_Cached'].map(extract_valid_encounters)


def compute_avg_encounter_frequency(valid_times):
    # Ensure the input is not empty and has at least two timestamps
    if valid_times is None or len(valid_times) < 2:
        return None

    # Convert the input to a pandas Series of datetime objects
    valid_times_series = pd.Series(pd.to_datetime(valid_times))

    # Calculate time differences in hours
    time_diffs = valid_times_series.diff().dt.total_seconds() / 3600  # Convert seconds to hours

    # Calculate and return the average time difference, excluding NaN
    return time_diffs[1:].mean()


# Apply the frequency calculation to all rows
df['Avg_Encounter_Frequency_Hours'] = df['Valid_Encounter_Times'].map(compute_avg_encounter_frequency)

# Display the updated DataFrame
print(df[['Row_Index', 'valid_previous_indices', 'Avg_Encounter_Frequency_Hours']].head())

print(df[['Row_Index', 'valid_previous_indices', 'Avg_Encounter_Frequency_Hours']].head(50))

# ----------------------------- #
# Step 14: Optimized First or Follow-Up Determination
# ----------------------------- #

# Sort the DataFrame by patient ID and scheduled time (SCHD_DTTM)
df = df.sort_values(by=['PT_ID', 'SCHD_DTTM']).reset_index(drop=True)

# Use a group-based transformation to label the first encounter as "First" and others as "Follow-Up"
df['First_Follow_Up'] = df.groupby('PT_ID').cumcount().apply(lambda x: 'First' if x == 0 else 'Follow-Up')

# Display the updated DataFrame with the new column
print(df[['Row_Index', 'PT_ID', 'SCHD_DTTM', 'First_Follow_Up']].head())


# ----------------------------- #
# Step 15: Optimized Last Appointment Status
# ----------------------------- #

# Define a function to extract the last appointment status
def extract_last_appointment_status(previous_rows):
    # If there are no valid previous rows, return NaN
    if not previous_rows:
        return pd.NA

    # Get the STATUS_CD of the most recent valid appointment
    return previous_rows[-1]['STATUS_CD']  # Last row represents the most recent appointment

# Compute the last appointment status in a vectorized manner
df['Last_Appointment_Status'] = df['Previous_Rows_Cached'].map(extract_last_appointment_status)

# Display the updated DataFrame with the new column
print(df[['Row_Index', 'valid_previous_indices', 'Last_Appointment_Status']].head(20))


# ----------------------------- #
# Step 16: Further Optimized Average Scheduled to Encounter Time
# ----------------------------- #

def calculate_avg_schd_encounter_time_faster(df):
    # Explode the 'Previous_Rows_Cached' column into separate rows
    exploded_rows = df.explode('Previous_Rows_Cached')

    # Extract SCHD_DTTM and ENCOUNTER_DTTM directly from the exploded column
    schd_dttm = pd.to_datetime(
        exploded_rows['Previous_Rows_Cached'].map(lambda r: r.get('SCHD_DTTM') if isinstance(r, dict) else None),
        errors='coerce'
    )
    encounter_dttm = pd.to_datetime(
        exploded_rows['Previous_Rows_Cached'].map(lambda r: r.get('ENCOUNTER_DTTM') if isinstance(r, dict) else None),
        errors='coerce'
    )

    # Calculate valid time differences (in hours) directly
    time_differences = (encounter_dttm - schd_dttm).dt.total_seconds() / 3600

    # Combine the valid time differences with the original index
    exploded_rows['time_difference_hours'] = time_differences

    # Group by the original index and calculate the mean time difference for each row
    avg_time_diff_per_row = (
        exploded_rows.groupby(exploded_rows.index)['time_difference_hours']
        .mean()
        .reindex(df.index)  # Align with the original DataFrame index
    )

    # Assign the calculated averages back to the original DataFrame
    df['Avg_Schd_Encounter_Diff_Hours'] = avg_time_diff_per_row

    return df

# Apply the faster function to calculate average scheduled to encounter time
df = calculate_avg_schd_encounter_time_faster(df)

# Display the updated dataframe with the new column
print(df[['Row_Index', 'valid_previous_indices', 'Avg_Schd_Encounter_Diff_Hours']].head())


# ----------------------------- #
# Step 17: Fully Optimized Average Arrival Lag
# ----------------------------- #

def calculate_average_lag_faster(df):
    # Explode the 'Previous_Rows_Cached' column into separate rows
    exploded_df = df.explode('Previous_Rows_Cached')

    # Extract ARRIVED_DTTM and ENCOUNTER_DTTM directly from the exploded data
    exploded_df['ARRIVED_DTTM'] = pd.to_datetime(
        exploded_df['Previous_Rows_Cached'].map(lambda r: r.get('ARRIVED_DTTM') if isinstance(r, dict) else None),
        errors='coerce'
    )
    exploded_df['ENCOUNTER_DTTM'] = pd.to_datetime(
        exploded_df['Previous_Rows_Cached'].map(lambda r: r.get('ENCOUNTER_DTTM') if isinstance(r, dict) else None),
        errors='coerce'
    )

    # Calculate time differences (in hours) where both ARRIVED_DTTM and ENCOUNTER_DTTM are valid
    exploded_df['time_difference_hours'] = (exploded_df['ARRIVED_DTTM'] - exploded_df['ENCOUNTER_DTTM']).dt.total_seconds() / 3600

    # Group by the original index and calculate the mean of time differences
    avg_time_diff_per_row = (
        exploded_df.groupby(exploded_df.index)['time_difference_hours']
        .mean()
        .reindex(df.index)  # Align with the original DataFrame
    )

    # Assign the result back to the original DataFrame
    df['Average_Arrival_Lag_Hours'] = avg_time_diff_per_row

    return df

# Apply the optimized function
df = calculate_average_lag_faster(df)

# Display the updated DataFrame
print(df[['Row_Index', 'valid_previous_indices', 'Average_Arrival_Lag_Hours']].head())


# ----------------------------- #
# Step 18: Fully Optimized Time Since Last Visit
# ----------------------------- #

# Pre-convert all SCHD_DTTM to datetime in bulk (if not already done)
df['SCHD_DTTM'] = pd.to_datetime(df['SCHD_DTTM'], errors='coerce')

# Explode the 'Previous_Rows_Cached' column into separate rows
exploded_df = df[['SCHD_DTTM', 'Previous_Rows_Cached']].explode('Previous_Rows_Cached')

# Extract valid ENCOUNTER_DTTM values directly from exploded data
exploded_df['ENCOUNTER_DTTM'] = pd.to_datetime(
    exploded_df['Previous_Rows_Cached'].map(
        lambda r: r.get('ENCOUNTER_DTTM') if isinstance(r, dict) and r.get('STATUS_CD') != 'Canceled' else None
    ),
    errors='coerce'
)

# Group by the original indices and calculate the most recent (max) valid ENCOUNTER_DTTM
last_valid_encounter = exploded_df.groupby(exploded_df.index)['ENCOUNTER_DTTM'].max()

# Align the calculated times with the original DataFrame and compute the time difference in hours
df['Time_Since_Last_Visit'] = (df['SCHD_DTTM'] - last_valid_encounter.reindex(df.index)).dt.total_seconds() / 3600

# Display the relevant columns for verification
print(df[['Row_Index', 'valid_previous_indices', 'Time_Since_Last_Visit']].head())


# ----------------------------- #
# Step 19: Optimized Cancellation Reason Category Mapping
# ----------------------------- #

# Flatten the dictionary to map each reason to its corresponding category
reason_to_category = {reason: category for category, reasons in reason_category_mapping.items() for reason in reasons}

# Explode the `Previous_Rows_Cached` column into separate rows for processing
exploded_df = df[['Row_Index', 'Previous_Rows_Cached']].explode('Previous_Rows_Cached')

# Extract STATUS_CD and CNCL_REASON_DESCR directly from the exploded rows
exploded_df['STATUS_CD'] = exploded_df['Previous_Rows_Cached'].map(
    lambda r: r.get('STATUS_CD') if isinstance(r, dict) else None
)
exploded_df['CNCL_REASON_DESCR'] = exploded_df['Previous_Rows_Cached'].map(
    lambda r: r.get('CNCL_REASON_DESCR') if isinstance(r, dict) else None
)

# Filter only rows where STATUS_CD == 'Canceled'
canceled_df = exploded_df[exploded_df['STATUS_CD'] == 'Canceled']

# Get the last cancellation reason for each original row (group by index)
last_canceled_reason = canceled_df.groupby(canceled_df.index)['CNCL_REASON_DESCR'].last()

# Map the last cancellation reason to its category
last_cancellation_category = last_canceled_reason.map(lambda reason: reason_to_category.get(reason, 'Unknown'))

# Assign the result back to the original DataFrame
df['Last_Cancellation_Category'] = df.index.map(last_cancellation_category)

# Display the updated DataFrame with the new column
print(df[['Row_Index', 'Last_Cancellation_Category']].head(30))



# Step 20: # Extract the last VISIT_TYPE and ENCOUNTER_TYPE using a faster approach
def extract_last_values(previous_rows_cached):
    if not isinstance(previous_rows_cached, list) or not previous_rows_cached:
        return None, None

    # Extract the last row
    last_row = previous_rows_cached[-1]

    # Extract VISIT_TYPE and ENCOUNTER_TYPE
    return last_row.get('VISIT_TYPE'), last_row.get('ENCOUNTER_TYPE')

# Apply the function using list comprehension
last_values = [
    extract_last_values(row['Previous_Rows_Cached']) if isinstance(row['Previous_Rows_Cached'], list) else (None, None)
    for _, row in df.iterrows()
]

# Convert the result into a DataFrame and assign new columns
last_values_df = pd.DataFrame(last_values, columns=['Last_VISIT_TYPE', 'Last_ENCOUNTER_TYPE'])

# Concatenate the new columns with the original DataFrame
df = pd.concat([df, last_values_df], axis=1)

# Display the updated DataFrame
print(df[['Row_Index', 'Last_VISIT_TYPE', 'Last_ENCOUNTER_TYPE']].head(30))


# Display the updated DataFrame
print(df[['Row_Index', 'Last_VISIT_TYPE', 'Last_ENCOUNTER_TYPE','valid_previous_indices','VISIT_TYPE', 'ENCOUNTER_TYPE' ]].head(30))

# Step 21: Function to check if there's any "Y" in PROV_INTERP_IND in the last valid indices
def check_prov_interp(row):
    previous_rows = row.get('Previous_Rows_Cached', [])

    # Check if previous_rows is a valid list
    if not isinstance(previous_rows, list):
        return None

    # Check if any "Y" exists in the PROV_INTERP_IND field
    for r in previous_rows:
        if r.get('PROV_INTERP_IND') == 'Y':
            return True

    # Return False if no "Y" is found
    return False

# Apply the function to create a new variable
df['Has_Prov_Interp_Y'] = df.apply(check_prov_interp, axis=1)

# Display the updated DataFrame with the new column
print(df[['Row_Index', 'Has_Prov_Interp_Y']].head(30))


# ----------------------------- #
# Step 19: Save Processed Data
# ----------------------------- #

# List of required columns to be saved
columns_to_save = [
    'PT_ID', 'ENCOUNTER_DTTM', 'VISIT_TYPE','ENCOUNTER_TYPE', 'STATUS_CD', 'SCHD_DTTM', 'CNCL_DTTM', 'CNCL_REASON_DESCR',
    'cancelled_within_24h', 'valid_previous_indices', 'Row_Index', 'Cancellation_Count', 'Patient Choice',
    'Clinic Management', 'Patient Health', 'Socioeconomic', 'Provider Choice', 'No_Show_Left_Count',
    'Provider_Change_Count', 'Provider_Changed_Last_Visit', 'Department_Change_Count',
    'Department_Changed_Last_Visit', 'Last_Visit_Duration', 'Avg_STD_DURATION', 'Rescheduled_Appointments_Count',
    'Encounter_Month', 'Encounter_Weekday', 'Weekend_Indicator', 'Encounter_Hour', 'Avg_Encounter_Frequency_Hours',
    'First_Follow_Up', 'Last_Appointment_Status', 'Avg_Schd_Encounter_Diff_Hours', 'Average_Arrival_Lag_Hours',
    'Time_Since_Last_Visit',
    'Has_Prov_Interp_Y',
    'Last_VISIT_TYPE', 'Last_ENCOUNTER_TYPE','Last_Cancellation_Category'
]

# Save the DataFrame with only the required columns (local)
csv_save_path = os.path.join(_script_dir, 'df_variables_0717.csv')
df[columns_to_save].to_csv(csv_save_path, index=False)


print(f"Number of rows after removing duplicates: {len(df)}")

# Print column names with one example value from each
for column in df.columns:
    print(f"{column}: {df[column].iloc[0]}")