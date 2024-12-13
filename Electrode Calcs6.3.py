import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Add this function to the top of your script, after imports
def smooth_data(data, window_length=7, polyorder=3):
    """Smooth data using Savitzky-Golay filter."""
    if len(data) < window_length:  # Ensure window length is valid
        return data
    return savgol_filter(data, window_length=window_length, polyorder=polyorder)

# Transformer tap changer data (example values)
tap_positions = [
    (1, 194.7), (2, 200.9), (3, 206.6), (4, 211.5), (5, 217.8), (6, 223.4),
    (7, 228), (8, 234), (9, 239.1), (10, 245.8), (11, 251.4), (12, 255.8),
    (13, 261.9), (14, 268.3), (15, 273.3), (16, 278.5), (17, 283.9),
    (18, 289.5), (19, 295.3), (20, 301.4), (21, 305.6), (22, 312.1),
    (23, 318.8), (24, 325.9), (25, 330.8), (26, 335.9), (27, 341.1)
]

# Resistance and reactance inputs
R_A, X_A = 2.50, 0.568  # Electrode A
R_B, X_B = 2.50, 0.568  # Electrode B
R_C, X_C = 2.50, 0.568  # Electrode C

# Transformer inputs
D9, E9 = 0.0, 0.46
D10, E10 = 0.0, 0.46
D11, E11 = 0.0, 0.46

# Resistance range
resistances = np.arange(0.0, 4.2, 0.1)
highlight_resistance = 2.0  # The resistance to highlight

# Lists for storing data for plotting
delta_active_powers = []
delta_electrode_currents = []
delta_resistances = []

star_active_powers = []
star_electrode_currents = []
star_resistances = []

highlight_delta_active_powers = []
highlight_delta_electrode_currents = []

highlight_star_active_powers = []
highlight_star_electrode_currents = []

# Loop through each tap position
for tap, voltage in tap_positions:
    V_AB, V_BC, V_CA = voltage, voltage, voltage  # Set all transformer secondary voltages

    for config in ["Delta", "Star"]:
        # Adjust voltages for star configuration
        if config == "Star":
            V_AB = voltage / np.sqrt(3)
            V_BC = voltage / np.sqrt(3)
            V_CA = voltage / np.sqrt(3)

        # Temporary storage for the current loop
        active_powers = []
        electrode_currents = []
        resistances_list = []

        for resistance in resistances:
            # Update resistances
            R_A, R_B, R_C = resistance, resistance, resistance

            # Construct Matrix1
            matrix1 = np.array([
                [D9 + R_A + R_B, -E9 - X_A - X_B, -R_B, X_B, -R_A, X_A],
                [E9 + X_A + X_B, D9 + R_A + R_B, -X_B, -R_B, -X_A, -R_A],
                [-R_B, X_B, D10 + R_B + R_C, -E10 - X_B - X_C, -R_C, X_C],
                [-X_B, -R_B, E10 + X_B + X_C, D10 + R_B + R_C, -X_C, -R_C],
                [-R_A, X_A, -R_C, X_C, D11 + R_C + R_A, -E11 - X_C - X_A],
                [-X_A, -R_A, -X_C, -R_C, E11 + X_C + X_A, D11 + R_C + R_A]
            ])

            # Add small diagonal regularization to prevent singular matrix
            matrix1 += np.eye(6) * 1e-6

            # Construct Matrix2
            matrix2 = np.array([
                [V_AB],
                [0.0],
                [-V_BC / 2],
                [V_BC * 0.866],
                [-V_CA / 2],
                [-V_CA * 0.866]
            ])

            # Handle singular matrix with try-except
            try:
                matrix1_inv = np.linalg.inv(matrix1)
            except np.linalg.LinAlgError:
                continue

            # Calculate matrix product
            matrix_product = np.dot(matrix1_inv, matrix2)
            T11, T12, T13, T14, T15, T16 = matrix_product.flatten()

            # Electrode current
            Elect_kA = np.sqrt((T11 - T15)**2 + (T12 - T16)**2)

            # Phase power
            Phase_MW = Elect_kA**2 * R_A * 0.001

            # Total active power
            Active_Power = Phase_MW * 3

            # Append to temporary lists
            active_powers.append(Active_Power)
            electrode_currents.append(Elect_kA)
            resistances_list.append(resistance)

            # Append to highlight lists if resistance matches
            if np.isclose(resistance, highlight_resistance):
                if config == "Delta":
                    highlight_delta_active_powers.append(Active_Power)
                    highlight_delta_electrode_currents.append(Elect_kA)
                else:
                    highlight_star_active_powers.append(Active_Power)
                    highlight_star_electrode_currents.append(Elect_kA)

        # Insert NaN between sequences to break the plot
        active_powers.append(np.nan)
        electrode_currents.append(np.nan)
        resistances_list.append(np.nan)

        # Append to main lists
        if config == "Delta":
            delta_active_powers.extend(active_powers)
            delta_electrode_currents.extend(electrode_currents)
            delta_resistances.extend(resistances_list)
            delta_electrode_currents_smooth = smooth_data(delta_electrode_currents, window_length=7, polyorder=3)
        else:
            star_active_powers.extend(active_powers)
            star_electrode_currents.extend(electrode_currents)
            star_resistances.extend(resistances_list)
            star_electrode_currents_smooth = smooth_data(star_electrode_currents, window_length=7, polyorder=3)

# Add MVA constraint curve
phases = 3  # Three-phase system
mva_limit = 48  # Operational MVA limit (MVA)
reactance = 0.00102  # Fixed reactance value (Ohms)
currents = np.linspace(20, 150, 100)  # Electrode current values in kA

mva_constraint_active_power = []
mva_constraint_currents = []

for current in currents:
    current_a = current * 1000  # Convert to A
    term = ((mva_limit * 1e6) / phases)**2 - (current_a**2 * reactance)**2
    if term < 0:
        continue
    mva_value = phases * np.sqrt(term)
    mva_constraint_active_power.append(mva_value / 1e6)  # Convert to MW
    mva_constraint_currents.append(current)

# Plotting Active Power, Electrode Current, and Resistance
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Delta configuration
ax1.plot(delta_active_powers, delta_electrode_currents, label='Delta Configuration', linestyle='-', color='green', linewidth=1)
for resistance in resistances:
    delta_indices = [i for i, res in enumerate(delta_resistances) if np.isclose(res, resistance)]
    ax1.plot(
        [delta_active_powers[i] for i in delta_indices],
        [delta_electrode_currents[i] for i in delta_indices],
        linestyle='-', linewidth=1, alpha=0.5
    )
ax1.plot(highlight_delta_active_powers, highlight_delta_electrode_currents, label='Delta 2.0 mOhm', color='red', linewidth=1)

# Plot Star configuration
ax1.plot(star_active_powers, star_electrode_currents, label='Star Configuration', linestyle='--', color='orange', linewidth=1)
for resistance in resistances:
    star_indices = [i for i, res in enumerate(star_resistances) if np.isclose(res, resistance)]
    ax1.plot(
        [star_active_powers[i] for i in star_indices],
        [star_electrode_currents[i] for i in star_indices],
        linestyle='--', linewidth=1, alpha=0.5
    )
ax1.plot(highlight_star_active_powers, highlight_star_electrode_currents, label='Star 2.0 mOhm', color='red', linewidth=1)

# Plot MVA constraint
ax1.plot(mva_constraint_active_power, mva_constraint_currents, label='MVA Constraint', linestyle='--', color='magenta', linewidth=2)

# Add Normal and Maximum Electrode Current lines
ax1.axhline(y=84.62, color='blue', linestyle='-', linewidth=1.0, label='Normal Electrode Current (84.62 kA)')
ax1.axhline(y=92.51, color='red', linestyle='-', linewidth=1.0, label='Maximum Electrode Current (92.51 kA)')

# Labels and limits
ax1.set_xlabel("Active Power (MW)")
ax1.set_ylabel("Electrode Current (kA)")
ax1.set_xlim(5, 50)
ax1.set_ylim(20, 150)
ax1.grid(True)
ax1.legend(loc='upper left', fontsize=8)

# Add title
x_ticks = np.arange(0, 50, 5)  # X-axis ticks from 0 to 55 with increments of 5 MW
y_ticks = np.arange(25, 125, 10)  # Y-axis ticks from 0 to 150 with increments of 10 kA

# Set the ticks
ax1.set_xticks(x_ticks)
ax1.set_yticks(y_ticks)

# Important data points (example values)
important_points = [
    (30, 84.62, "Normal Electr Crnt"),  # Example: (Active Power, Electrode Current, Label)
    (40, 92.51, "Max Electr Crnt"),
    (10, 41, "2 mOhm"),  # Add more points as needed
    (42.24, 84.62, "Design Point"),  # Add more points as needed
]

# Annotate the important points with coordinate values
for x, y, label in important_points:
    ax1.scatter(x, y, color='red', zorder=5)  # Add a red marker at the point
    ax1.annotate(
        f"{label}\n({x}, {y:.2f})",  # Include the coordinate values in the label
        xy=(x, y),  # Point coordinates
        xytext=(x + 2, y + 5),  # Offset text position slightly
        textcoords='offset points',
        arrowprops=dict(facecolor='black', arrowstyle='->'),  # Add an arrow
        fontsize=8, color='blue', bbox=dict(boxstyle="round,pad=0.3", edgecolor='blue', facecolor='white')
    )

# Optional: Customize tick labels (e.g., add units or format them)
ax1.set_xticklabels([f"{x} MW" for x in x_ticks])
ax1.set_yticklabels([f"{y} kA" for y in y_ticks])
plt.title("M1 Furnace PVI Diagram 3 x 16 MVA Transformers, X = 1.02mΩ")
plt.show()

# Placeholder for storing segmented data for each tap position
delta_active_power_per_tap = []
star_active_power_per_tap = []
resistance_per_tap = []

# Separate data for each tap position
num_resistances = len(resistances) + 1  # Including the NaN separator
for i in range(0, len(delta_active_powers), num_resistances):
    delta_active_power_per_tap.append(smooth_data(delta_active_powers[i:i + num_resistances - 1], window_length=7, polyorder=3))
    star_active_power_per_tap.append(smooth_data(star_active_powers[i:i + num_resistances - 1], window_length=7, polyorder=3))
    resistance_per_tap.append(resistances)

# Plotting Active Power vs Resistance for all tap positions
fig, ax1 = plt.subplots(figsize=(14, 8))

# Plot Delta and Star Configurations
for tap_index, tap in enumerate(tap_positions):
    ax1.plot(
        resistance_per_tap[tap_index],
        delta_active_power_per_tap[tap_index],
        linestyle='-', color='green', linewidth=1.0, alpha=0.5, label=f'Tap {tap[0]} Delta' if tap_index == 0 else None
    )
    ax1.plot(
        resistance_per_tap[tap_index],
        star_active_power_per_tap[tap_index],
        linestyle='--', color='orange', linewidth=1.0, alpha=0.5, label=f'Tap {tap[0]} Star' if tap_index == 0 else None
    )

# Convert the tap_positions list to a dictionary for MVA Constraints
tap_positions_dict = {f"Tap {tap[0]}": tap[1] for tap in tap_positions}

# Resistance range (mOhm)
resistances = np.arange(0.0, 4.2, 0.2)  # Furnace resistances

# Transformer parameters
reactance = 0.00102  # Transformer reactance (Ohms)
mva_constraints = [48, 48]  # MVA limits for dashed red curves

# Plot additional MVA constraint data
for mva_limit in mva_constraints:
    constraint_curve = []
    for resistance_mOhm in resistances:
        resistance = resistance_mOhm / 1000  # Convert to Ohms
        impedance = np.sqrt(resistance**2 + reactance**2)
        voltage = max(tap_positions_dict.values())  # Highest voltage tap
        current = mva_limit * 1000 / (np.sqrt(3) * voltage)
        mw = mva_limit * (resistance / impedance)  # Active Power
        constraint_curve.append(mw)
    ax1.plot(
        resistances,
        constraint_curve,
        linestyle='-',
        linewidth=2,
        label=f'{mva_limit} MVA Constraint',
        color='red'
    )

# Define electrode currents in kA (visible in the attachment)
electrode_currents = [84.62, 92.31, 100.0]  # kA

# Calculate and plot electrode current trends
for current in electrode_currents:
    current_mw_trend = []  # Active power (MW) for this current
    for resistance_mOhm in resistances:
        resistance = resistance_mOhm / 1000  # Convert to Ohms
        active_power = 3 * (current**2) * resistance  # Active Power (MW)
        current_mw_trend.append(active_power)

    # Plot the trend
    ax1.plot(
        resistances, 
        current_mw_trend, 
        linestyle='-', 
        linewidth=1, 
        color='blue', 
        label=f'Elec I = {current} kA'
    )

# Configure the plot
ax1.set_xlabel("Furnace Resistance (mOhm)")
ax1.set_ylabel("Active Power (MW)")
ax1.set_xlim(min(resistances), max(resistances))
ax1.set_ylim(0, max(max(delta_active_power_per_tap, key=np.max).max(), max(star_active_power_per_tap, key=np.max).max()) * 1.1)
ax1.grid(True)
ax1.legend(loc='upper right', fontsize=8)

# Add a legend for the blue trends
plt.legend(loc="upper left", fontsize=8)

# Add Title and Labels (reuse existing ones)
plt.title("Furnace Resistance vs Active Power with Electrode Current Trends")
plt.xlabel("Furnace Resistance (mOhm)")
plt.ylabel("Active Power (MW)")
plt.grid(True)
plt.xlim(0, 3.5)
plt.ylim(0, 60)

# Add title
x_ticks = np.arange(0, 4, 0.5)  # X-axis ticks from 0 to 55 with increments of 5 MW
y_ticks = np.arange(0, 70, 5)  # Y-axis ticks from 0 to 150 with increments of 10 kA

# Set the ticks
ax1.set_xticks(x_ticks)
ax1.set_yticks(y_ticks)

# Important data points (kA)
important_points = [
    (2.75, 60.0, "84.62 kA",),    # Example data point for current
    (3.0, 77.0, "92.31 kA",),    # Example data point for another current
    (3.1, 93.0, "100 kA",)     # Example for a higher current
]

# Annotate the important points with coordinate values
for x, y, label in important_points:
    ax1.annotate(
        label,  # Label text
        xy=(x, y),  # Point coordinates (text will align with the trend line)
        xytext=(x + 0.01, y),  # Adjust the text slightly to prevent overlap
        textcoords='data',
        fontsize=9, color='blue',  # Set text properties
       bbox=dict(boxstyle="round,pad=0.3", edgecolor='blue', facecolor='white', alpha=0.8) # Optional background
    )
    
# Important data points (MW, MVA)
important_points = [
    (2.0, 43.0, "43.0 MW"),  # Example data point for active power
    (2.5, 44.0, "48 MVA"),   # Example data point for MVA
   ]

# Annotate the important points with coordinate values
for x, y, label in important_points:
    ax1.annotate(
        label,  # Label text
        xy=(x, y),  # Point coordinates (text will align with the trend line)
        xytext=(x + 0.01, y),  # Adjust the text slightly to prevent overlap
        textcoords='data',
        fontsize=9, color='red',  # Set text properties
       bbox=dict(boxstyle="round,pad=0.3", edgecolor='red', facecolor='white', alpha=0.8) # Optional background
    )

# Tap positions and labels
tap_positions = [
    (0.5, 25, "Tap 1"),  # Example tap point
    (0.5, 30, "Tap 5"),
    (0.5, 54, "Tap 10"),
    (0.5, 45, "Tap 15"),
    (0.5, 62, "Tap 20"),
    (0.5, 80, "Tap 27")   # Add as many taps as necessary
]

# Annotate tap numbers directly on the graph
for x, y, label in tap_positions:
    ax1.annotate(
        label,  # Tap number text
        xy=(x, y),  # Coordinates for the tap point
        xytext=(x + 0.1, y + 2),  # Offset slightly for better readability
        textcoords='data',
        fontsize=9, color='green',  # Set text properties
        bbox=dict(boxstyle="round,pad=0.3", edgecolor='green', facecolor='white', alpha=0.8)  # Optional background
    )
    
# Set grid, title, and labels
plt.title("Furnace Resistance vs Active Power with Electrode Current Trends")
plt.xlabel("Furnace Resistance (mOhm)")
plt.ylabel("Active Power (MW)")
plt.grid(True)
plt.xlim(0, 3.5)
plt.ylim(0, 100)  # Adjusted to accommodate new points
    
# Optional: Customize tick labels (e.g., add units or format them)
ax1.set_xticklabels([f"{x} mOhm" for x in x_ticks])
ax1.set_yticklabels([f"{y} MW" for y in y_ticks])
plt.title("M1 Furnace PVI Diagram 3 x 16 MVA Transformers, X = ?mΩ")
plt.show()