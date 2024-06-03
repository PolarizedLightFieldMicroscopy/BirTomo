import time

def measure_computation_time(compute_function, *args, repetitions=1000):
    start_time = time.perf_counter()
    for _ in range(repetitions):
        compute_function(*args)
    elapsed_time = (time.perf_counter() - start_time) / repetitions
    return elapsed_time
