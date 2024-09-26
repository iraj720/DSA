import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Simulation Parameters
OPERATIONAL_AREA = 0.5  # square miles
MIN_TX_TX_SEPARATION = 10  # meters
TX_RX_SEPARATION_RANGE = (10, 100)  # meters
FREQUENCY_SHIFT = 1  # MHz
POWER_MARGIN_THRESHOLD = 3  # dB
NUM_LINKS = 20  # Adjust this to simulate different network sizes
NUM_TRIALS = 1  # Number of simulation runs

# Constants
SPEED_OF_LIGHT = 3e8  # m/s

# M-Sequence Codes (Simplified Representation)
AVAILABLE_CODES = [f"Code_{i}" for i in range(1, 11)]  # 10 unique codes

# Helper Functions
def path_loss(distances):
    """Calculate path loss using a free-space path loss model."""
    frequency = 2.4e9  # Hz (Assuming operation in the 2.4 GHz band)
    pl = 20 * np.log10(distances + 1e-6) + 20 * np.log10(frequency) + 20 * np.log10(4 * np.pi / SPEED_OF_LIGHT)
    return pl  # in dB

def cross_correlation_matrix(codes):
    """Calculate cross-correlation matrix between codes."""
    num_codes = len(codes)
    corr_matrix = np.full((num_codes, num_codes), 0.1)  # Default low cross-correlation
    for i in range(num_codes):
        corr_matrix[i, i] = 1.0  # Maximum interference for same code
    return corr_matrix

# Main Algorithm
def spectrum_deconfliction(positions):
    """Implement the Spectrum Deconfliction algorithm with M-sequence codes."""
    num_devices = len(positions)
    tx_positions = np.array([tx for tx, _ in positions])
    rx_positions = np.array([rx for _, rx in positions])

    # Assign M-sequence codes
    assigned_codes = []
    available_codes = AVAILABLE_CODES.copy()
    for n in range(num_devices):
        if available_codes:
            code_n = available_codes.pop(0)
        else:
            code_n = random.choice(AVAILABLE_CODES)
        assigned_codes.append(code_n)

    assigned_codes = np.array(assigned_codes)
    code_corr_matrix = cross_correlation_matrix(assigned_codes)

    # Initialize frequency channels and powers
    frequency_channels = np.zeros(num_devices)
    tx_powers = np.zeros(num_devices)
    channels = [0]

    # Precompute distance matrices
    tx_to_rx_dist = np.linalg.norm(tx_positions[:, np.newaxis] - rx_positions[np.newaxis, :], axis=2)
    tx_to_rx_pl = path_loss(tx_to_rx_dist)

    # Start deconfliction
    for n in range(num_devices):
        # Compatibility score and interference margins
        ct_score = 0
        total_tp_linear = 0
        max_pm = 0

        # Interference from existing transmitters to new receiver
        if n > 0:
            distances = np.linalg.norm(tx_positions[:n] - rx_positions[n], axis=1)
            path_losses = path_loss(distances)
            code_corrs = code_corr_matrix[:n, n]
            interference_powers = -path_losses + 10 * np.log10(code_corrs)
            interference_powers_linear = 10 ** (interference_powers / 10)
            total_tp_linear = interference_powers_linear.sum()
            total_tp_db = 10 * np.log10(total_tp_linear + 1e-12)

            # Compatibility check
            compatible = interference_powers < -100  # Threshold
            ct_score += compatible.sum()

            # Aggregate interference check
            if total_tp_db > -100:
                # Change frequency and recompute CTs
                channel = max(channels) + FREQUENCY_SHIFT
                channels.append(channel)
                frequency_channels[n] = channel
                continue

        # Interference from new transmitter to existing receivers
        if n > 0:
            distances = np.linalg.norm(tx_positions[n] - rx_positions[:n], axis=1)
            path_losses = path_loss(distances)
            code_corrs = code_corr_matrix[n, :n]
            interference_powers = -path_losses + 10 * np.log10(code_corrs)
            curr_pm = interference_powers - (-100)
            max_pm = np.maximum(max_pm, curr_pm.max())
            compatible = interference_powers < -100  # Threshold
            ct_score += compatible.sum()

        if ct_score == 2 * n:
            # Compatible with all devices
            frequency_channels[n] = channels[0]
        elif max_pm <= POWER_MARGIN_THRESHOLD:
            # Adjust Tx power
            tx_powers[n] = max_pm
            frequency_channels[n] = channels[0]
        else:
            # Change frequency
            channel = max(channels) + FREQUENCY_SHIFT
            channels.append(channel)
            frequency_channels[n] = channel

    return frequency_channels, tx_powers, assigned_codes

# Simulation Execution
def run_simulation():
    positions = generate_positions(NUM_LINKS)
    frequency_channels, tx_powers, assigned_codes = spectrum_deconfliction(positions)

    # Output results
    for i in range(NUM_LINKS):
        print(f"Link {i+1}:")
        print(f"  Tx Position: {positions[i][0]}")
        print(f"  Rx Position: {positions[i][1]}")
        print(f"  Assigned Frequency Channel: {frequency_channels[i]:.1f} MHz")
        print(f"  Assigned Code: {assigned_codes[i]}")
        print(f"  Tx Power Adjustment: {tx_powers[i]:.2f} dB\n")

    # Visualizations
    plot_network(positions, frequency_channels, assigned_codes)
    plot_frequency_allocation(frequency_channels, assigned_codes)

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

def plot_network(positions, frequency_channels, assigned_codes):
    """Plot the network topology and frequency channel assignments."""
    plt.figure(figsize=(10, 10))
    unique_channels = np.unique(frequency_channels)
    colors = plt.cm.get_cmap('tab20', len(unique_channels))

    for i, ((tx_pos, rx_pos), channel, code) in enumerate(zip(positions, frequency_channels, assigned_codes)):
        color_idx = np.where(unique_channels == channel)[0][0]
        plt.plot([tx_pos[0], rx_pos[0]], [tx_pos[1], rx_pos[1]], color=colors(color_idx),
                 label=f"Link {i+1} (Ch {channel:.1f} MHz, {code})")
        plt.scatter(tx_pos[0], tx_pos[1], marker='^', color=colors(color_idx))
        plt.scatter(rx_pos[0], rx_pos[1], marker='s', color=colors(color_idx))

    plt.title("Spectrum Deconfliction Network Topology with M-Sequence Codes")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    # Create legend with unique entries
    handles = [mpatches.Patch(color=colors(i), label=f"Channel {ch:.1f} MHz") for i, ch in enumerate(unique_channels)]
    plt.legend(handles=handles, loc='upper right')
    plt.grid(True)
    plt.show()

def plot_frequency_allocation(frequency_channels, assigned_codes):
    """Plot the frequency spectrum allocation map with code assignments."""
    plt.figure(figsize=(10, 2))
    num_channels = len(np.unique(frequency_channels))
    freq_min = frequency_channels.min()
    freq_max = frequency_channels.max() + FREQUENCY_SHIFT  # Assuming channels are continuous

    # Create a horizontal bar to represent the spectrum
    for idx, ch in enumerate(sorted(np.unique(frequency_channels))):
        links_on_channel = [i+1 for i, fc in enumerate(frequency_channels) if fc == ch]
        codes_on_channel = [assigned_codes[i] for i, fc in enumerate(frequency_channels) if fc == ch]
        plt.barh(0, FREQUENCY_SHIFT, left=ch, height=0.5, align='edge', edgecolor='black', color='skyblue')
        plt.text(ch + FREQUENCY_SHIFT/2, 0.25,
                 f"Ch {ch:.1f} MHz\nLinks: {', '.join(map(str, links_on_channel))}\nCodes: {', '.join(codes_on_channel)}",
                 ha='center', va='center', fontsize=8)

    plt.title("Frequency Spectrum Allocation Map with M-Sequence Codes")
    plt.xlabel("Frequency (MHz)")
    plt.yticks([])
    plt.ylim(-0.1, 0.6)
    plt.xlim(freq_min - FREQUENCY_SHIFT, freq_max + FREQUENCY_SHIFT)
    plt.grid(axis='x')
    plt.show()

# Run the simulation
run_simulation()
