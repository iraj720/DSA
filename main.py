import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Simulation Parameters
OPERATIONAL_AREA = 0.5  # square miles
MIN_TX_TX_SEPARATION = 10  # meters
TX_RX_SEPARATION_RANGE = (10, 100)  # meters
BANDWIDTH_RANGE = (1, 5)  # MHz, variable bandwidth per link
POWER_MARGIN_THRESHOLD = 3  # dB
NUM_LINKS = 20  # Number of links

# Constants
SPEED_OF_LIGHT = 3e8  # m/s
FREQUENCY_MIN = 1000  # MHz, minimum frequency
FREQUENCY_MAX = 1010  # MHz, maximum frequency
FREQUENCY_STEP = 0.1  # MHz, step size for frequency segments

# M-Sequence Codes (Simplified Representation)
AVAILABLE_CODES = [f"Code_{i}" for i in range(1, 6)]  # 5 unique codes

# Helper Functions
def path_loss(distance):
    """Calculate path loss using a free-space path loss model."""
    frequency = 2.4e9  # Hz (Assuming operation in the 2.4 GHz band)
    pl = 20 * np.log10(distance + 1e-6) + 20 * np.log10(frequency) + 20 * np.log10(4 * np.pi / SPEED_OF_LIGHT)
    return pl  # in dB

def cross_correlation(code1, code2):
    """Calculate cross-correlation between two codes."""
    if code1 == code2:
        return 1.0  # Maximum interference
    else:
        return 0.2  # Assume low cross-correlation for different codes

def generate_positions(num_links):
    """Generate random positions for Tx/Rx pairs."""
    area_side = np.sqrt(OPERATIONAL_AREA * 2.59e6)  # Convert square miles to square meters
    positions = []
    tx_positions = []
    for _ in range(num_links):
        while True:
            tx_pos = np.array([random.uniform(0, area_side), random.uniform(0, area_side)])
            rx_distance = random.uniform(*TX_RX_SEPARATION_RANGE)
            angle = random.uniform(0, 2 * np.pi)
            rx_pos = tx_pos + rx_distance * np.array([np.cos(angle), np.sin(angle)])
            # Ensure minimum Tx-Tx separation
            if len(tx_positions) == 0 or np.all(np.linalg.norm(tx_positions - tx_pos, axis=1) >= MIN_TX_TX_SEPARATION):
                positions.append((tx_pos, rx_pos))
                tx_positions.append(tx_pos)
                break
    return positions

def spectrum_deconfliction(positions, bandwidths):
    """Implement the Spectrum Deconfliction algorithm with variable start frequencies and optimization."""
    num_devices = len(positions)
    tx_positions = np.array([tx for tx, _ in positions])
    rx_positions = np.array([rx for _, rx in positions])

    # Assign initial frequencies randomly within the available spectrum
    assigned_frequencies = np.array([random.uniform(FREQUENCY_MIN, FREQUENCY_MAX - bw) for bw in bandwidths])
    assigned_codes = [random.choice(AVAILABLE_CODES) for _ in range(num_devices)]

    # Divide the spectrum into 0.1 MHz segments
    frequency_segments = np.arange(FREQUENCY_MIN, FREQUENCY_MAX + FREQUENCY_STEP, FREQUENCY_STEP)

    # Optimization loop
    max_iterations = 100
    for iteration in range(max_iterations):
        interference_matrix = np.zeros((num_devices, num_devices))
        total_interference = np.zeros(len(frequency_segments))
        changes_made = False
       
        # Compute interference and cross-correlation
        for i in range(num_devices):
            # each Link reaquest comes here
            freq_i_start = assigned_frequencies[i]
            freq_i_end = freq_i_start + bandwidths[i]
            idx_i_start = int((freq_i_start - FREQUENCY_MIN) / FREQUENCY_STEP)
            idx_i_end = int((freq_i_end - FREQUENCY_MIN) / FREQUENCY_STEP)
            segments_i = set(range(idx_i_start, idx_i_end))

            for j in range(i + 1, num_devices):
                freq_j_start = assigned_frequencies[j]
                freq_j_end = freq_j_start + bandwidths[j]
                idx_j_start = int((freq_j_start - FREQUENCY_MIN) / FREQUENCY_STEP)
                idx_j_end = int((freq_j_end - FREQUENCY_MIN) / FREQUENCY_STEP)
                segments_j = set(range(idx_j_start, idx_j_end))

                overlapping_segments = segments_i & segments_j
                if overlapping_segments:
                    # Calculate interference between link i and link j
                    d_tx_i_rx_j = np.linalg.norm(tx_positions[i] - rx_positions[j])
                    d_tx_j_rx_i = np.linalg.norm(tx_positions[j] - rx_positions[i])
                    pl_tx_i_rx_j = path_loss(d_tx_i_rx_j)
                    pl_tx_j_rx_i = path_loss(d_tx_j_rx_i)
                    code_corr = cross_correlation(assigned_codes[i], assigned_codes[j])

                    interference_power_i_j = -pl_tx_i_rx_j + 10 * np.log10(code_corr)
                    interference_power_j_i = -pl_tx_j_rx_i + 10 * np.log10(code_corr)

                    # Sum interference over overlapping segments
                    for seg in overlapping_segments:
                        total_interference[seg] += 10 ** (interference_power_i_j / 10)
                        total_interference[seg] += 10 ** (interference_power_j_i / 10)

                    # If interference exceeds threshold, adjust frequencies or codes
                    if interference_power_i_j >= -100 or interference_power_j_i >= -100:
                        # Adjust frequencies
                        new_freq_i = assigned_frequencies[i] + random.uniform(-1, 1)
                        new_freq_j = assigned_frequencies[j] + random.uniform(-1, 1)

                        # Ensure frequencies are within bounds
                        new_freq_i = max(FREQUENCY_MIN, min(new_freq_i, FREQUENCY_MAX - bandwidths[i]))
                        new_freq_j = max(FREQUENCY_MIN, min(new_freq_j, FREQUENCY_MAX - bandwidths[j]))

                        assigned_frequencies[i] = new_freq_i
                        assigned_frequencies[j] = new_freq_j

                        # Optionally, change codes
                        available_codes_i = [code for code in AVAILABLE_CODES if code != assigned_codes[i]]
                        available_codes_j = [code for code in AVAILABLE_CODES if code != assigned_codes[j]]
                        if available_codes_i:
                            assigned_codes[i] = random.choice(available_codes_i)
                        if available_codes_j:
                            assigned_codes[j] = random.choice(available_codes_j)

                        changes_made = True

        if not changes_made:
            break  # Optimization converged

    # Prepare frequency allocations for visualization
    frequency_allocations = []
    for i in range(num_devices):
        freq_start = assigned_frequencies[i]
        freq_end = freq_start + bandwidths[i]
        frequency_allocations.append((freq_start, freq_end))

    return frequency_allocations, assigned_codes, total_interference

def plot_network(positions, assigned_codes):
    """Plot the network topology and code assignments."""
    plt.figure(figsize=(10, 10))
    for i, ((tx_pos, rx_pos), code) in enumerate(zip(positions, assigned_codes)):
        plt.plot([tx_pos[0], rx_pos[0]], [tx_pos[1], rx_pos[1]], label=f"Link {i+1} ({code})")
        plt.scatter(tx_pos[0], tx_pos[1], marker='^')
        plt.scatter(rx_pos[0], rx_pos[1], marker='s')
    plt.title("Network Topology with Code Assignments")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
    plt.grid(True)
    plt.show()

def plot_frequency_allocation(frequency_allocations, assigned_codes, bandwidths):
    """Plot the frequency spectrum allocation map with overlaps."""
    plt.figure(figsize=(12, 4))
    freq_min = FREQUENCY_MIN
    freq_max = FREQUENCY_MAX

    # Create a horizontal bar to represent the spectrum
    plt.barh(0, freq_max - freq_min, left=freq_min, height=0.5, align='edge',
             edgecolor='black', color='lightgrey', zorder=1)

    # Plot each link's frequency allocation
    y_offset = 0.5  # Vertical offset for text labels
    for idx, (freq_alloc, code, bw) in enumerate(zip(frequency_allocations, assigned_codes, bandwidths)):
        bar = plt.barh(0, freq_alloc[1] - freq_alloc[0], left=freq_alloc[0], height=0.5, align='edge',
                       edgecolor='black', alpha=0.7, label=f"Link {idx+1}", zorder=2)
        # Adjust text position to avoid overlap
        text_x = freq_alloc[0] + (freq_alloc[1] - freq_alloc[0]) / 2
        text_y = 0.25 + (idx % 2) * y_offset * 0.1  # Stagger text vertically
        plt.text(text_x, text_y,
                 f"L{idx+1}, {code}\n{bw:.2f} MHz",
                 ha='center', va='bottom', fontsize=8, rotation=45, zorder=3)

    plt.title("Frequency Spectrum Allocation with Variable Start Frequencies")
    plt.xlabel("Frequency (MHz)")
    plt.yticks([])
    plt.ylim(-0.1, 0.8)
    plt.xlim(freq_min, freq_max)
    plt.grid(axis='x')
    plt.tight_layout()
    plt.show()

def run_simulation():
    positions = generate_positions(NUM_LINKS)
    bandwidths = np.random.uniform(BANDWIDTH_RANGE[0], BANDWIDTH_RANGE[1], NUM_LINKS)
    frequency_allocations, assigned_codes, total_interference = spectrum_deconfliction(positions, bandwidths)

    # Output results
    for i in range(NUM_LINKS):
        print(f"Link {i+1}:")
        print(f"  Tx Position: {positions[i][0]}")
        print(f"  Rx Position: {positions[i][1]}")
        print(f"  Bandwidth Requirement: {bandwidths[i]:.2f} MHz")
        freq_alloc = frequency_allocations[i]
        print(f"  Assigned Frequency: {freq_alloc[0]:.2f} MHz to {freq_alloc[1]:.2f} MHz")
        print(f"  Assigned Code: {assigned_codes[i]}")
        print()

    # Visualizations
    plot_network(positions, assigned_codes)
    plot_frequency_allocation(frequency_allocations, assigned_codes, bandwidths)

    # Plot total interference over frequency
    plt.figure(figsize=(12, 4))
    freq_segments = np.arange(FREQUENCY_MIN, FREQUENCY_MAX + FREQUENCY_STEP, FREQUENCY_STEP)
    plt.plot(freq_segments[:len(total_interference)], 10 * np.log10(total_interference + 1e-12))
    plt.title("Total Interference Over Frequency")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Interference Power (dB)")
    plt.grid(True)
    plt.show()

# Run the simulation
run_simulation()
