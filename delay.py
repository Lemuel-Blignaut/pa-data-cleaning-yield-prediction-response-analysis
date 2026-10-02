import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2


def manual_lag_correction(input_df,time_col, time_lag, logging_interval):
    
    stabilised_df = input_df.copy()
    stabilised_df[time_col] = pd.to_datetime(stabilised_df[time_col])
    stabilised_df = stabilised_df.sort_values(by=time_col).reset_index(drop=True)
 
    tot_points = len(stabilised_df)
    points_per_timestamp = int(tot_points / len(stabilised_df[time_col].unique()))
    print(f"Total points: {tot_points}, Points per timestamp: {points_per_timestamp}")

 
    # 4. Calculate the Shift Index
    # Shift = Lag Time / Logging Interval
    shift_steps = int(round(time_lag / logging_interval))*points_per_timestamp

    # 5. Apply the Shift
    # We shift the 'yield' and 'moisture' values BACKWARDS in the sequence
    # so they align with the GPS coordinates where the header actually was
    col_list = stabilised_df.columns.tolist()  # Get the list of columns
    col_list.remove('geometry')  # Remove geometry from the list to shift only data columns
    shifted_df = stabilised_df.copy()
    for col in col_list:
        shifted_df[col] = shifted_df[col].shift(-shift_steps)
    
    shifted_df = shifted_df.dropna(subset=['time'])  # Drop rows where any of the shifted columns are NaN   
        
    return shifted_df


def prepare_shifted_data(df, delay, time_col=None, time_type='index', yield_col='Yield', polling_rate=1.0):
    """
    Sorts the dataframe chronologically and shifts the specified yield column temporally 
    to simulate correcting for a specific sensor delay, accounting for the logging/polling rate.
    Handles missing GPS coordinates safely.
    
    Parameters:
    - df: pandas DataFrame containing the yield data.
    - delay: Integer or Float representing the target delay time in seconds to apply.
    - time_col: String, the name of the timestamp column (if time_type='timestamp').
    - time_type: String, either 'timestamp' (uses time_col) or 'index' (uses row order).
    - yield_col: String, the name of the column containing the yield values.
    - polling_rate: Float, the time in seconds between each logged row.
    
    Returns:
    - A new pandas DataFrame with the shifted yield column and all NaNs (yield or GPS) dropped.
    """
    df_shifted = df.copy()
    
    # 1. Ensure Chronological Order First
    if time_type == 'timestamp' and time_col is not None:
        df_shifted = df_shifted.sort_values(time_col).reset_index(drop=True)
    elif time_type == 'index':
        df_shifted = df_shifted.sort_index().reset_index(drop=True)
    else:
        raise ValueError("time_type must be 'timestamp' (requires time_col) or 'index'.")
        
    # 2. Apply the Temporal Shift
    # Convert the target delay (in seconds) to the discrete number of rows to shift.
    row_shift = int(round(delay / polling_rate))
    
    # Shift the yield column backward (negative shift moves future yield to current coordinates)
    df_shifted[yield_col] = df_shifted[yield_col].shift(-row_shift)
    
    # 3. Clean up Missing Data AFTER Shifting
    # This ensures we don't break the chronological row spacing before the shift happens.
    # We drop rows where Yield, UTM_X, or UTM_Y are missing.
    for col in [yield_col, 'UTM_X', 'UTM_Y']:
        if col  in df_shifted.columns:
            df_shifted = df_shifted.dropna(subset=col)
    
    return df_shifted


def run_pcdi_analysis(df, header_width, delay_min=-32, delay_max=32, 
                      time_col=None, time_type='index', yield_col='Yield', 
                      polling_rate=1.0, plot=False):
    """
    Executes a single iteration of the Phase Correlation Delay Identification (PCDI) method[cite: 7, 8].
    
    Parameters:
    - df: pandas DataFrame containing the yield data and UTM coordinates.
    - header_width: Float, the width of the combine harvester header.
    - delay_min: Integer, the minimum delay time to test.
    - delay_max: Integer, the maximum delay time to test.
    - time_col: String, name of the timestamp column.
    - time_type: String, 'timestamp' or 'index'.
    - yield_col: String, name of the yield column.
    - polling_rate: Float, time in seconds between each logged row.
    - plot: Boolean, if True, plots the Relative Spatial Consistency (RSC) curve.
    
    Returns:
    - best_delay: Integer, the delay time that produced the highest RSC.
    - results_df: pandas DataFrame containing the RSC value for every tested delay.
    """
    # 1. Setup Raster Constants
    # Pixel size roughly 50% to 85% of header width ensures adjacent passes fall into adjacent pixels[cite: 333, 334].
    pixel_size = header_width * 0.7  
    
    df = df.copy()
    # Drop NaNs temporarily just to calculate the spatial bounding box safely
    df['UTM_X'] = df.geometry.x
    df['UTM_Y'] = df.geometry.y
    
    df_bounds = df.dropna(subset=['UTM_X', 'UTM_Y'])
    x_min, y_min = df_bounds['UTM_X'].min(), df_bounds['UTM_Y'].min()
    x_range = df_bounds['UTM_X'].max() - x_min
    y_range = df_bounds['UTM_Y'].max() - y_min
    max_dim = max(x_range, y_range)
    
    # Calculate rasterization size N as the next even power of 2 for FFT
    n_pixels = int(2**np.ceil(np.log2(max_dim / pixel_size)))
    
    # 2. Pre-generate Gaussian White Noise (GWN)
    # Replaces missing data points to prevent spatial relationships in missing data from skewing results[cite: 140, 141].
    yield_mean = df_bounds[yield_col].mean()
    yield_std = df_bounds[yield_col].std()
    gwn_grid = np.random.normal(yield_mean, yield_std, (n_pixels, n_pixels))
    
    rsc_results = []

    # 3. Iterate through the defined delay range
    for d in range(delay_min, delay_max + 1):
        # Shift data and clean missing points
        temp_df = prepare_shifted_data(df, d, time_col, time_type, yield_col, polling_rate)
        
        # Calculate pixel indices for each UTM coordinate
        x_idx = ((temp_df['UTM_X'] - x_min) / pixel_size).astype(int).clip(0, n_pixels-1)
        y_idx = ((temp_df['UTM_Y'] - y_min) / pixel_size).astype(int).clip(0, n_pixels-1)
        
        temp_df['x_idx'] = x_idx
        temp_df['y_idx'] = y_idx
        # Average multiple yield values that fall into the exact same pixel [cite: 128]
        grouped = temp_df.groupby(['x_idx', 'y_idx'])[yield_col].mean().reset_index()
        
        grid = np.zeros((n_pixels, n_pixels))
        valid_mask = np.zeros((n_pixels, n_pixels), dtype=bool)
        
        grid[grouped['x_idx'], grouped['y_idx']] = grouped[yield_col]
        valid_mask[grouped['x_idx'], grouped['y_idx']] = True
        
        # 4. Pixel Coupling and Translation Offset
        # Shift mask diagonally by (1,1) to compare adjacent harvest transects [cite: 337, 341]
        shifted_mask = np.roll(valid_mask, shift=(1, 1), axis=(0, 1))
        
        # Apply GWN to uncoupled pixels to minimize their impact [cite: 259]
        g1 = np.where(valid_mask & shifted_mask, grid, gwn_grid)
        g2 = np.roll(g1, shift=(1, 1), axis=(0, 1))
        
        # 5. Phase Correlation using Fast Fourier Transform [cite: 347]
        f1 = fft2(g1)
        f2 = fft2(g2)
        
        # Calculate cross-power spectrum [cite: 80, 81]
        num = f1 * np.conj(f2)
        den = np.abs(num) + 1e-12 # Prevent division by zero
        cps = num / den
        
        # Inverse FFT to obtain the phase correlation matrix P [cite: 84, 85]
        p_matrix = np.real(ifft2(cps))
        
        # The Relative Spatial Consistency (RSC) index is located at c_(0,0) [cite: 223]
        rsc = p_matrix[0, 0]
        rsc_results.append({'Delay': d, 'RSC': rsc})

    results_df = pd.DataFrame(rsc_results)
    best_delay = results_df.loc[results_df['RSC'].idxmax(), 'Delay']
    
    # 6. Visualization (Individual Iteration)
    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(results_df['Delay'], results_df['RSC'], marker='o', linestyle='-', color='b')
        max_rsc = results_df['RSC'].max()
        plt.axvline(x=best_delay, color='r', linestyle='--', label=f'Best Delay: {best_delay}s')
        plt.scatter([best_delay], [max_rsc], color='red', s=100, zorder=5)
        plt.title('Individual Run: RSC vs. Delay Time')
        plt.xlabel('Delay Time (seconds)')
        plt.ylabel('Normalized RSC ($c_{0,0}$)')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    return best_delay, results_df


def run_robust_pcdi(df, header_width, delay_min=-32, delay_max=32, iterations=10,
                    time_col=None, time_type='index', yield_col='Yield', polling_rate=1.0,
                    plot_individual=False, plot_final=True, plot_map=True):
    """
    Runs PCDI multiple times to account for GWN randomness, averages results, 
    and plots spatial yield maps before and after correction[cite: 570, 571].
    
    Parameters:
    - (Standard parameters from run_pcdi_analysis)
    - iterations: Integer, number of algorithm runs (default 10)[cite: 414].
    - plot_individual: Boolean, plots RSC curve for every iteration.
    - plot_final: Boolean, plots final averaged RSC curve.
    - plot_map: Boolean, plots side-by-side geographic maps of unshifted vs. shifted data.
    """
    print(f"Running {iterations} iterations of PCDI testing delays from {delay_min}s to {delay_max}s...")
    
    all_rsc_curves = []
    best_delays = []
    
    for i in range(iterations):
        print(f"  -> Processing Iteration {i+1}/{iterations}...")
        best_d, results_df = run_pcdi_analysis(
            df, header_width, delay_min, delay_max, 
            time_col, time_type, yield_col, polling_rate, plot_individual
        )
        all_rsc_curves.append(results_df['RSC'].values)
        best_delays.append(best_d)
        
    # Stack curves and calculate average RSC
    stacked_curves = np.vstack(all_rsc_curves)
    avg_rsc_curve = np.mean(stacked_curves, axis=0)
    delay_std = np.std(best_delays)
    
    final_results = pd.DataFrame({
        'Delay': results_df['Delay'].values, 
        'Avg_RSC': avg_rsc_curve
    })
    
    overall_best_delay = final_results.loc[final_results['Avg_RSC'].idxmax(), 'Delay']
    
    print("\n--- Final Results ---")
    print(f"Optimal Delay: {overall_best_delay} seconds")
    print(f"Standard Deviation: {delay_std:.2f} seconds")
    
    if delay_std > 3.0:
        print("WARNING: High standard deviation (>3.0s) detected. The delay estimate may be unreliable[cite: 573, 574].")
        
    # Visualization: Final Averaged RSC Curve
    if plot_final:
        plt.figure(figsize=(10, 6))
        plt.plot(final_results['Delay'], final_results['Avg_RSC'], marker='o', linestyle='-', color='g', label=f'Average RSC (n={iterations})')
        max_rsc = final_results['Avg_RSC'].max()
        plt.axvline(x=overall_best_delay, color='r', linestyle='--', label=f'Best Overall Delay: {overall_best_delay}s')
        plt.scatter([overall_best_delay], [max_rsc], color='red', s=100, zorder=5)
        plt.title(f'Robust PCDI Analysis: Averaged RSC vs. Delay Time')
        plt.xlabel('Delay Time (seconds)')
        plt.ylabel('Average Normalized RSC')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    # Visualization: Side-by-Side Spatial Map Comparison
    if plot_map:
        # Prepare original data (0 delay) and corrected data (best delay)
        original_df = prepare_shifted_data(df, 0, time_col, time_type, yield_col, polling_rate)
        corrected_df = prepare_shifted_data(df, overall_best_delay, time_col, time_type, yield_col, polling_rate)
        
        # Determine consistent color scale boundaries for both maps
        vmin = min(original_df[yield_col].quantile(0.05), corrected_df[yield_col].quantile(0.05))
        vmax = max(original_df[yield_col].quantile(0.95), corrected_df[yield_col].quantile(0.95))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
        
        # Plot Original Map
        sc1 = ax1.scatter(original_df['UTM_X'], original_df['UTM_Y'], 
                          c=original_df[yield_col], cmap='YlGn', s=10, vmin=vmin, vmax=vmax)
        ax1.set_title('Original Unshifted Yield Map', fontsize=14)
        ax1.set_xlabel('UTM X')
        ax1.set_ylabel('UTM Y')
        ax1.grid(True, linestyle=':', alpha=0.5)
        
        # Plot Corrected Map
        sc2 = ax2.scatter(corrected_df['UTM_X'], corrected_df['UTM_Y'], 
                          c=corrected_df[yield_col], cmap='YlGn', s=10, vmin=vmin, vmax=vmax)
        ax2.set_title(f'Corrected Yield Map ({overall_best_delay}s shift)', fontsize=14)
        ax2.set_xlabel('UTM X')
        ax2.grid(True, linestyle=':', alpha=0.5)
        
        # Add a shared colorbar
        cbar = fig.colorbar(sc1, ax=[ax1, ax2], fraction=0.03, pad=0.04)
        cbar.set_label(f'Yield ({yield_col})', rotation=270, labelpad=15)
        
        plt.suptitle('Spatial Yield Comparison Before and After PCDI Correction', fontsize=16)
        plt.show()
    
    return overall_best_delay, final_results, delay_std


