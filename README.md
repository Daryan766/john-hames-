# john-james-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Load CSV File Directly ===
csv_path = "/Users/u5638980/Downloads/fig3dataA.csv"
spot_data = pd.read_csv(csv_path, encoding='latin1')

# Column name of data to plot
idx_y = "MEDIAN_INTENSITY_CH1"

# Make a copied subset of spot_data
try:
    track_data = spot_data[["TRACK_ID", "POSITION_T", idx_y]].copy()
except KeyError as e:
    raise KeyError(f"Required column missing in CSV: {e}")

# Safely remove first 3 rows (assuming metadata)
track_data = track_data.iloc[3:].copy()

# Convert to numeric and drop bad rows
track_data[["TRACK_ID", "POSITION_T", idx_y]] = track_data[["TRACK_ID", "POSITION_T", idx_y]].apply(pd.to_numeric, errors='coerce')
track_data.dropna(subset=["TRACK_ID", "POSITION_T", idx_y], inplace=True)

# Sort data and reset index
track_data = track_data.sort_values(by=["TRACK_ID", "POSITION_T"]).reset_index(drop=True)

# Time step calculation from a sample track
first_track_id = track_data["TRACK_ID"].iloc[0]
example_track = track_data[track_data["TRACK_ID"] == first_track_id]
if len(example_track) > 1:
    time_step = example_track["POSITION_T"].iloc[1] - example_track["POSITION_T"].iloc[0]
else:
    raise ValueError("Not enough points to calculate time step from first track.")

# === Auto-set minimum track length ===
track_lengths = track_data.groupby("TRACK_ID").size()
min_track_length = int(track_lengths.quantile(0.25))  # 25th percentile
min_track_length = max(min_track_length, 10)  # enforce a floor
print(f"Auto-set min_track_length = {min_track_length}")

# Filter valid tracks
valid_tracks = []
filtered_data = []

for track_id, group in track_data.groupby("TRACK_ID"):
    if len(group) >= min_track_length:
        valid_tracks.append(track_id)
        filtered_data.append(group)

# Check for valid tracks
if not filtered_data:
    raise ValueError("No valid tracks meet the minimum length.")

# Combine filtered data
track_data = pd.concat(filtered_data)

# Max time
max_track = track_data["POSITION_T"].max()

# Create output matrix
output_table = np.full((int(max_track / time_step) + 1, len(valid_tracks)), np.nan)

# Fill matrix
for i, track_id in enumerate(valid_tracks):
    this_track = track_data[track_data["TRACK_ID"] == track_id]
    time_points = (this_track["POSITION_T"] / time_step).astype(int)
    values = this_track[idx_y].values
    output_table[time_points, i] = values

# Time values for x-axis
x_vals = np.linspace(0, max_track, output_table.shape[0])

# Plotting
plt.figure(figsize=(10, 10))
for i in range(output_table.shape[1]):
    plt.plot(x_vals, output_table[:, i], alpha=0.3, color='gray', linewidth=1)

# Plot mean trace
mean_data = np.nanmean(output_table, axis=1)
plt.plot(x_vals, mean_data, linewidth=3, color='red', label='Mean')

# Final tweaks
plt.xlim([0, max_track])
plt.xticks(np.arange(0, max_track + 1, 60))
plt.xlabel('Time')
plt.ylabel(idx_y)
plt.title(f"Tracks of {idx_y}")
plt.legend()
plt.tight_layout()
plt.show()

# Export mean trace to CSV
mean_df = pd.DataFrame({
    "Time": x_vals,
    "Mean_" + idx_y: mean_data
})
mean_df.to_csv("mean_trace_output.csv", index=False)
print("✅ Mean trace exported to 'mean_trace_output.csv'")
