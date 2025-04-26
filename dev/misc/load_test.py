# %% [markdown]
# # CPU load test
#
# This script is used to test the CPU load of the container.

# %%
import time
import math
import multiprocessing
import threading
import psutil # For monitoring
import numpy as np
from collections import defaultdict
import statistics

# %%
def cpu_load(duration_seconds=60):
    """Generates CPU load for a specified duration."""
    # print(f"Starting CPU load task on core {multiprocessing.current_process().name} for {duration_seconds} seconds...")
    start_time = time.time()
    while time.time() - start_time < duration_seconds:
        # Perform some calculations to consume CPU
        _ = math.sqrt(time.time()) * math.sin(time.time()) * math.exp(math.log(time.time() + 1))
    # print(f"CPU load task on core {multiprocessing.current_process().name} finished.")

# %%
def monitor_stats(stop_event, results_list, interval=1):
    """Monitors CPU and memory usage periodically."""
    print(f"Starting monitoring with {interval}s interval.")
    psutil.cpu_percent(interval=None, percpu=True) # Call once initially to clear internal state
    time.sleep(0.1) # Small delay before first real reading
    while not stop_event.is_set():
        start_read_time = time.time()
        timestamp = start_read_time
        cpu_percents = psutil.cpu_percent(interval=0, percpu=True) # Non-blocking, compares since last call
        mem_percent = psutil.virtual_memory().percent
        results_list.append({
            "timestamp": timestamp,
            "cpu_percent_per_core": cpu_percents,
            "memory_percent": mem_percent
        })
        # Wait for the next interval, checking stop_event frequently
        # Adjust sleep time based on how long psutil took
        elapsed_read_time = time.time() - start_read_time
        sleep_time = max(0, interval - elapsed_read_time)
        stop_event.wait(sleep_time)
    print("Monitoring stopped.")

# %%
def run_load_on_cores(num_cores, duration_seconds=60):
    """Runs the cpu_load function on multiple cores and monitors stats."""
    processes = []
    monitor_results = []
    stop_monitoring_event = threading.Event()

    # Start the monitoring thread
    monitor_thread = threading.Thread(
        target=monitor_stats,
        args=(stop_monitoring_event, monitor_results, 1), # Monitor every 1 second
        daemon=True # Allows program to exit even if this thread is alive
    )
    monitor_thread.start()

    print(f"\n>>> Starting load test on {num_cores} cores for {duration_seconds} seconds. <<<")
    start_time = time.time()
    for i in range(num_cores):
        # Give processes names for easier identification (optional)
        p = multiprocessing.Process(target=cpu_load, args=(duration_seconds,), name=f"LoadCore-{i}")
        p.start()
        processes.append(p)

    # Wait for all load processes to complete
    for p in processes:
        p.join()

    end_time = time.time()
    actual_duration = end_time - start_time
    print(f"Load test on {num_cores} cores finished in {actual_duration:.2f} seconds.")

    # Stop the monitoring thread
    stop_monitoring_event.set()
    monitor_thread.join(timeout=5) # Wait max 5s for monitor thread

    # Process and print statistics
    if monitor_results:
        print("\n--- Performance Statistics ---")
        # Calculate average/max CPU per core and overall memory
        avg_cpu_per_core = defaultdict(list)
        max_cpu_per_core = defaultdict(float)
        avg_mem = []

        # Filter results to only include those within the actual test duration
        relevant_results = [r for r in monitor_results if start_time <= r['timestamp'] <= end_time]
        if not relevant_results:
             print("No monitoring data collected during the test execution window.")
             return

        num_physical_cores = len(relevant_results[0]['cpu_percent_per_core']) # Get core count from first reading

        for reading in relevant_results:
            avg_mem.append(reading['memory_percent'])
            # Ensure we don't try to access more cores than reported by psutil
            cores_in_reading = len(reading['cpu_percent_per_core'])
            for i in range(min(num_physical_cores, cores_in_reading)):
                core_load = reading['cpu_percent_per_core'][i]
                avg_cpu_per_core[i].append(core_load)
                max_cpu_per_core[i] = max(max_cpu_per_core[i], core_load)

        print(f"Monitoring Samples: {len(relevant_results)}")
        print(f"Average Memory Usage: {statistics.mean(avg_mem):.2f}%" if avg_mem else "N/A")
        print("CPU Usage per Core (Average / Max):")
        total_avg_cpu = 0
        for i in range(num_physical_cores):
            if i in avg_cpu_per_core and avg_cpu_per_core[i]:
                 avg = statistics.mean(avg_cpu_per_core[i])
                 max_val = max_cpu_per_core[i]
                 total_avg_cpu += avg
                 print(f"  Core {i}: {avg:.2f}% / {max_val:.2f}%")
            else:
                 print(f"  Core {i}: No data") # Should not happen normally
        print(f"Total Average CPU Usage (Sum of core averages): {total_avg_cpu:.2f}%")
        print(f"Overall Average CPU Usage (Across all cores): {total_avg_cpu / num_physical_cores:.2f}%")

    else:
        print("No monitoring data collected.")

# %%
def numpy_matrix_multiplication(size=100):
    """Performs matrix multiplication using numpy."""
    # Create two random matrices
    start_time = time.time()
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    C = np.dot(A, B)
    end_time = time.time()
    print(f"Numpy matrix multiplication of {size}x{size} matrix took {end_time - start_time:.2f} seconds")


# %%
if __name__ == "__main__":
    # Option 1: Run for a specific number of cores (e.g., all available)
    cores_to_test = multiprocessing.cpu_count() # Use all available cores by default
    test_duration = 30 # seconds
    print(f"Detected {cores_to_test} CPU cores available to the OS.")
    run_load_on_cores(cores_to_test, test_duration)

    # Option 2: Iterate through different core counts (uncomment to use)
    # max_cores = multiprocessing.cpu_count()
    # test_duration = 15 # Shorter duration for iterative testing
    # print(f"Detected {max_cores} CPU cores available to the OS.")
    # for i in range(1, max_cores + 1):
    #     run_load_on_cores(i, test_duration)
    #     time.sleep(2) # Pause between tests



# %%
if __name__ == "__main__":
    # Option 2: Execute numpy matrix multiplication test
    numpy_matrix_multiplication(20000)
