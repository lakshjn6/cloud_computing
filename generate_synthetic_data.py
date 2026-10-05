import numpy as np
import pandas as pd

def generate_bitbrains_cloud_workload(
    num_rows=2000,
    sampling_interval_sec=300,
    output_filepath="cloud_workload_dataset.csv",
    seed=42
):
    """
    Generates a realistic synthetic cloud workload dataset matching Bitbrains/Azure trace schema.
    
    Columns produced:
    - Timestamp [ms]
    - CPU cores
    - CPU capacity provisioned [MHZ]
    - CPU usage [MHZ]
    - CPU usage [%]
    - Memory capacity provisioned [KB]
    - Memory usage [KB]
    - Disk read throughput [KB/s]
    - Disk write throughput [KB/s]
    - Network received throughput [KB/s]
    - Network transmitted throughput [KB/s]
    """
    np.random.seed(seed)
    
    # Base timestamp (similar to 1376314846 in Bitbrains trace)
    start_timestamp_sec = 1376314846
    timestamps_ms = [(start_timestamp_sec + i * sampling_interval_sec) for i in range(num_rows)]
    
    t = np.arange(num_rows)
    
    # 1. Diurnal cycle (24-hour periodicity with 5-minute sampling: 288 steps per day)
    steps_per_day = 288
    diurnal = 35 + 25 * np.sin(2 * np.pi * (t - 48) / steps_per_day)
    
    # 2. Spikes (bursty workload batch jobs)
    spike_prob = 0.03
    raw_spikes = np.random.binomial(1, spike_prob, size=num_rows) * np.random.uniform(20, 45, size=num_rows)
    smoothed_spikes = np.zeros(num_rows)
    curr = 0
    for i in range(num_rows):
        curr = curr * 0.75 + raw_spikes[i]
        smoothed_spikes[i] = curr
        
    # 3. Noise process (AR(1))
    noise = np.zeros(num_rows)
    phi = 0.8
    for i in range(1, num_rows):
        noise[i] = phi * noise[i-1] + np.random.normal(0, 2.5)
        
    # Compute CPU usage (%)
    raw_cpu_pct = diurnal + smoothed_spikes + noise
    cpu_pct = np.clip(raw_cpu_pct, 0.5, 98.0)
    
    # Compute CPU Provisioned (MHz) & Usage (MHz)
    cpu_cores = np.full(num_rows, 4)
    cpu_capacity_mhz = np.full(num_rows, 11703.99824)
    cpu_usage_mhz = (cpu_pct / 100.0) * cpu_capacity_mhz + np.random.normal(0, 5.0, num_rows)
    cpu_usage_mhz = np.clip(cpu_usage_mhz, 50.0, 11703.99824)
    
    # Re-calculate exact CPU usage [%] to maintain absolute mathematical consistency
    cpu_usage_pct = np.round((cpu_usage_mhz / cpu_capacity_mhz) * 100.0, 8)
    
    # Memory capacity provisioned & Usage in KB (6.71E+07 KB = 67.1 GB provisioned)
    mem_capacity_kb = np.full(num_rows, 67108864.0)
    # Memory usage shows retention / plateauing dynamics relative to CPU
    mem_base = 6000000.0
    mem_lag = np.roll(cpu_usage_mhz, 4)
    mem_lag[:4] = cpu_usage_mhz[:4]
    mem_usage_kb = mem_base + (mem_lag / 11703.99824) * 20000000.0 + np.random.normal(0, 150000.0, num_rows)
    # Periodic spikes in memory
    mem_usage_kb += (smoothed_spikes / 100.0) * 10000000.0
    mem_usage_kb = np.clip(mem_usage_kb, 0.0, mem_capacity_kb * 0.95)
    
    # Throughput metrics
    disk_read = np.clip(np.random.exponential(1.5, num_rows) + smoothed_spikes * 0.2, 0, 500)
    disk_write = np.clip(np.random.exponential(15.0, num_rows) + (cpu_pct / 100.0) * 5000, 0, 25000)
    net_rx = np.clip((cpu_pct / 100.0) * 800 * (np.random.uniform(0.1, 1.2, num_rows)), 0, 2000)
    net_tx = np.clip(np.random.exponential(2.0, num_rows) + (cpu_pct / 100.0) * 50, 0, 200)

    df = pd.DataFrame({
        "Timestamp [ms]": timestamps_ms,
        "CPU cores": cpu_cores,
        "CPU capacity provisioned [MHZ]": np.round(cpu_capacity_mhz, 5),
        "CPU usage [MHZ]": np.round(cpu_usage_mhz, 5),
        "CPU usage [%]": cpu_usage_pct,
        "Memory capacity provisioned [KB]": mem_capacity_kb,
        "Memory usage [KB]": np.round(mem_usage_kb, 1),
        "Disk read throughput [KB/s]": np.round(disk_read, 5),
        "Disk write throughput [KB/s]": np.round(disk_write, 5),
        "Network received throughput [KB/s]": np.round(net_rx, 5),
        "Network transmitted throughput [KB/s]": np.round(net_tx, 5)
    })
    
    df.to_csv(output_filepath, index=False)
    print(f"✅ Generated dataset matching Bitbrains trace format with {len(df)} records at '{output_filepath}'.")
    return df

if __name__ == "__main__":
    generate_bitbrains_cloud_workload()
