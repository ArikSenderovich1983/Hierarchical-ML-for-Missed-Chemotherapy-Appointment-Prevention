# ==============================
#  TABLE OF CONTENTS
# ==============================

# 1. Import Libraries
# 2. Mount Google Drive
# 3. Read & Combine Text Files
# 4. Filter by Visit Types
# 5. Convert Columns to Datetime
# 6. Check & Remove Duplicates
# 7. Sort & Calculate Time Differences
# 8. Filter Patients with Multiple Appointments
# 9. Group by Patient & Process History
# 10. Add Valid Previous Indices
# 11. Save Final DataFrame
# 12. Extra Duplicate Checks



# =====================================================
# Step 1: Import Required Libraries
# =====================================================
import glob  # To find file paths matching a pattern
import pandas as pd  # To handle DataFrame operations

# =====================================================
# Step 2: Local paths (no Google Drive)
# =====================================================
import os
_script_dir = os.path.dirname(os.path.abspath(__file__))
_data_dir = os.path.join(_script_dir, "Merged", "Merged")

# =====================================================
# Step 3: Read and Combine All Text Files
# =====================================================
# Define the file path pattern to locate the text files
file_path_pattern = os.path.join(_data_dir, "*.txt")

# Use glob to get a list of all files matching the pattern
all_files = glob.glob(file_path_pattern)

# Initialize an empty list to hold individual DataFrames
dfs = []

# Loop over the list of files and read each one into a DataFrame
for file in all_files:
    # Read each file with '|' as the delimiter and append it to the list
    df = pd.read_csv(file, delimiter='|')
    dfs.append(df)

# Concatenate all DataFrames into a single DataFrame
merged_df = pd.concat(dfs, ignore_index=True)



# =====================================================
# Step 4: Filter Rows by Desired Visit Types
# =====================================================
# Define the list of visit types to include
desired_visit_types = [
    'BLOOD DRAW', 'ESTABLISHED PATIENT', 'ESTABLISHED PATIENT ON TX',
    'TREATMENT CHEMO', 'TREATMENT', 'NEW CHEMO', 'BLOOD PRODUCT'
]

# Filter the DataFrame to include only the desired visit types
df = merged_df[merged_df['VISIT_TYPE'].isin(desired_visit_types)]

# =====================================================
# Step 5: Convert Columns to Datetime Format
# =====================================================
# List of datetime columns to be converted
datetime_columns = [
    'ENCOUNTER_DT', 'ENCOUNTER_DTTM', 'ARRIVED_DTTM',
    'SCHD_DTTM', 'CNCL_DTTM', 'RESCHD_DTTM'
]

# Convert each column to datetime format, coercing errors to NaT
for col in datetime_columns:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Display DataFrame info and a preview to confirm changes
df.head()

total_rows = len(df)
print(f"Total number of rows in the DataFrame: {total_rows}")

# Check duplicated data
import pandas as pd

# Assuming df is your DataFrame

# Step 1: Identify duplicate groups
duplicate_groups = (
    df.groupby(['PT_ID', 'ENCOUNTER_DTTM', 'SCHD_DTTM'])
    .size()  # Count the occurrences for each group
    .reset_index(name='Count')  # Convert groupby result to a DataFrame
    .query('Count > 1')  # Filter to only include groups with duplicates
)

# Step 2: Filter the original DataFrame to include only rows in duplicate groups
duplicates = df.merge(
    duplicate_groups[['PT_ID', 'ENCOUNTER_DTTM', 'SCHD_DTTM']],
    on=['PT_ID', 'ENCOUNTER_DTTM', 'SCHD_DTTM'],
    how='inner'
)

# Step 3: Get the first 20 groups of duplicates
# Extract the unique duplicate groups and limit to 20
first_20_groups = duplicate_groups.head(20)[['PT_ID', 'ENCOUNTER_DTTM', 'SCHD_DTTM']]

# Filter full data for these 20 duplicate groups
duplicates_for_20_groups = duplicates.merge(first_20_groups, on=['PT_ID', 'ENCOUNTER_DTTM', 'SCHD_DTTM'], how='inner')

print(duplicates_for_20_groups)


import pandas as pd

# Assuming df is your DataFrame

# Step 1: Identify duplicate rows based on PT_ID and ENCOUNTER_DTTM
duplicate_counts = (
    df.groupby(['PT_ID', 'ENCOUNTER_DTTM', 'SCHD_DTTM'])
    .size()  # Count occurrences in each group
    .reset_index(name='Count')  # Convert groupby result to a DataFrame
)

# Step 2: Calculate the sum of duplicates (only count rows beyond the first instance in each group)
total_duplicates = duplicate_counts.query('Count > 1')['Count'].sum() - len(duplicate_counts.query('Count > 1'))

print(f"Total number of duplicate rows: {total_duplicates}")


# Remove duplicated data
# Step 3: Remove duplicate rows while keeping the first occurrence
df = df.drop_duplicates(subset=['PT_ID', 'ENCOUNTER_DTTM', 'SCHD_DTTM'], keep='first')

# Step 4: Optional: Print or save the cleaned DataFrame
print(f"Number of rows after removing duplicates: {len(df)}")



# =====================================================
# Step 6: Sort and Calculate Time Differences
# =====================================================
# Sort DataFrame by Patient ID and Encounter DateTime
df = df.sort_values(by=['PT_ID', 'ENCOUNTER_DTTM'])

# Calculate time difference between ENCOUNTER_DTTM and CNCL_DTTM
df['time_difference_hours'] = (df['ENCOUNTER_DTTM'] - df['CNCL_DTTM']).dt.total_seconds() / 3600

# Create a binary column for cancellations within 24 hours
df['cancelled_within_24h'] = df['time_difference_hours'].apply(lambda x: 1 if 0 <= x <= 24 else 0)

# Display rows where cancellation happened within 24 hours
cancelled_within_24h_df = df[df['cancelled_within_24h'] == 1]
cancelled_within_24h_df[['PT_ID', 'ENCOUNTER_DTTM', 'CNCL_DTTM', 'time_difference_hours', 'cancelled_within_24h']].head()






# =====================================================
# Step 7: Filter Patients with Multiple Appointments
# =====================================================
# Count the number of appointments per patient
appointment_counts = df['PT_ID'].value_counts()

# Get patient IDs with more than one appointment
patients_with_multiple_appointments = appointment_counts[appointment_counts > 1].index

# Filter the DataFrame to include only these patients
df = df[df['PT_ID'].isin(patients_with_multiple_appointments)]

# Display the new number of rows after filtering
filter_len = df.shape[0]
print(f"Number of rows after filtering patients with multiple appointments: {filter_len}")

# =====================================================
# Step 8: Group by Patient and Process History
# =====================================================
# Sort DataFrame by Patient ID and Scheduled DateTime
df = df.sort_values(by=['PT_ID', 'SCHD_DTTM']).reset_index(drop=True)

# Get the total number of unique patients
unique_patients = df['PT_ID'].nunique()
print(f"Total number of unique patients: {unique_patients}")

# Initialize a dictionary to store indices for valid previous rows
filtered_previous_indices = {}

# Group the DataFrame by Patient ID
grouped_patients = df.groupby('PT_ID')

# Process each patient group independently
for patient_idx, (patient_id, group) in enumerate(grouped_patients):
    num_appointments = group.shape[0]
    if (patient_idx + 1) % 5000 == 0 or patient_idx == 0:
        print(f"Processing patient {patient_idx + 1}/{unique_patients}...")

    for i, current_row in group.iterrows():
        current_schd_dttm = current_row['SCHD_DTTM']
        previous_rows = group[group['SCHD_DTTM'] < current_schd_dttm]
        valid_previous_rows = previous_rows[(~previous_rows['CNCL_DTTM'].isna() & (previous_rows['CNCL_DTTM'] < current_schd_dttm)) |
                                           (~previous_rows['ENCOUNTER_DTTM'].isna() & (previous_rows['ENCOUNTER_DTTM'] < current_schd_dttm))]
        filtered_previous_indices[i] = valid_previous_rows.index.tolist()


# =====================================================
# Add valid_previous_indices as a column to df
# =====================================================

# Convert the filtered_previous_indices dictionary to a Series
# Aligning indices from the dictionary to the original DataFrame
valid_indices_series = pd.Series(filtered_previous_indices)

# Add the new column to the DataFrame
df['valid_previous_indices'] = df.index.map(valid_indices_series)

# Verify the new column
print("New column 'valid_previous_indices' added to the DataFrame:")
print(df[['PT_ID', 'SCHD_DTTM', 'valid_previous_indices']].head(20))


# Verify the new column
print("New column 'valid_previous_indices' added to the DataFrame:")
print(df[['PT_ID', 'SCHD_DTTM','ENCOUNTER_DTTM','CNCL_DTTM','STATUS_CD', 'valid_previous_indices']].head(50))

# =====================================================
# Step 9: Save the Resulting DataFrame
# =====================================================
# Define the CSV save path (local)
csv_save_path = os.path.join(_script_dir, "df_with_valid_indices_0717.csv")

# Save the DataFrame to the specified path
df.to_csv(csv_save_path, index=False)

print(f"DataFrame saved to {csv_save_path}")

import pandas as pd

# Assuming df is your DataFrame

# Step 1: Identify duplicate rows based on PT_ID and ENCOUNTER_DTTM
duplicate_counts = (
    df.groupby(['PT_ID', 'ENCOUNTER_DTTM'])
    .size()  # Count occurrences in each group
    .reset_index(name='Count')  # Convert groupby result to a DataFrame
)

# Step 2: Calculate the sum of duplicates (only count rows beyond the first instance in each group)
total_duplicates = duplicate_counts.query('Count > 1')['Count'].sum() - len(duplicate_counts.query('Count > 1'))

print(f"Total number of duplicate rows: {total_duplicates}")

import pandas as pd

# Assuming df is your DataFrame

# Step 1: Identify duplicate groups
duplicate_groups = (
    df.groupby(['PT_ID', 'ENCOUNTER_DTTM'])
    .size()  # Count the occurrences for each group
    .reset_index(name='Count')  # Convert groupby result to a DataFrame
    .query('Count > 1')  # Filter to only include groups with duplicates
)

# Step 2: Filter the original DataFrame to include only rows in duplicate groups
duplicates = df.merge(
    duplicate_groups[['PT_ID', 'ENCOUNTER_DTTM']],
    on=['PT_ID', 'ENCOUNTER_DTTM'],
    how='inner'
)

# Step 3: Get the first 20 groups of duplicates
# Extract the unique duplicate groups and limit to 20
first_20_groups = duplicate_groups.head(20)[['PT_ID', 'ENCOUNTER_DTTM']]

# Filter full data for these 20 duplicate groups
duplicates_for_20_groups = duplicates.merge(first_20_groups, on=['PT_ID', 'ENCOUNTER_DTTM'], how='inner')

print(duplicates_for_20_groups)