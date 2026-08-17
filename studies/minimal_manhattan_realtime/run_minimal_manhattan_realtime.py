import os
import sys
import multiprocessing as mp

# Add root FleetPy path to sys.path
fleetpy_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if fleetpy_path not in sys.path:
    sys.path.append(fleetpy_path)

from run_scenarios import run_scenarios

if __name__ == "__main__":
    mp.freeze_support()
    
    study_dir = os.path.dirname(os.path.abspath(__file__))
    scenarios_dir = os.path.join(study_dir, "scenarios")
    
    constant_config = os.path.join(scenarios_dir, "constant_config.yaml")
    scenario_config = os.path.join(scenarios_dir, "scenario_config.csv")
    
    print("Running minimal Manhattan Realtime Ride-Pooling scenario...")
    run_scenarios(
        constant_config,
        scenario_config,
        log_level="info",
        n_cpu_per_sim=1,
        n_parallel_sim=1
    )
    print("\nRealtime simulation complete! Results saved in studies/minimal_manhattan_realtime/results/")
