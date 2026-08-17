# Minimal Manhattan Realtime V2 Ride-Pooling Study

This study defines a self-contained real-time ride-pooling simulation using the Manhattan benchmark dataset in FleetPy, leveraging the **`RealtimeV2Simulation`** environment with decoupled Fleet Simulation and Fleet Control threads.

## 📂 Study Directory Layout

```
studies/minimal_manhattan_realtime_v2/
├── README.md                              # Study documentation & workflow guide
├── run_minimal_manhattan_realtime_v2.py   # Real-time V2 simulation execution script
└── scenarios/
    ├── constant_config.yaml               # Environment (`sim_env: RealtimeV2Simulation`), network, & realtime parameters
    └── scenario_config.csv                # Scenario definition (`minimal_manhattan_realtime_v2_sc_1`)
```

---

## 📥 Dataset Requirements

This study utilizes the same Manhattan dataset as `studies/minimal_manhattan_realtime/`. If you haven't downloaded it yet, run:

```bash
python studies/minimal_manhattan_realtime/download_data.py
```

---

## 🚀 Execution & Results

### 1. Run Realtime V2 Simulation:
```bash
python studies/minimal_manhattan_realtime_v2/run_minimal_manhattan_realtime_v2.py
```

Real-time simulation results will be written to `studies/minimal_manhattan_realtime_v2/results/minimal_manhattan_realtime_v2_sc_1/`:
- `standard_eval.csv` (Aggregated KPIs: wait times, modal split, vehicle miles, pooled trip share)
- `1_user-stats.csv` (Per-trip passenger stats)
- `2-0_op-stats.csv` (Operator vehicle task logs)
- `rt_request_metrics.csv` (Per-request enqueue, process, and response times = Queue Wait + Optimization Time)
- `rt_tick_metrics.csv` (Per-tick duration for the unblocked fleet simulation loop)
- `rt_optimization_metrics.csv` (Dedicated metrics for each asynchronous batch optimization cycle)
- `rt_summary.csv` (Summary of response times, tick durations, and solver optimization performance)

### 2. Replay Visualization:
To visually replay vehicle movements and operational metrics:
```bash
python replay_pyplot.py studies/minimal_manhattan_realtime_v2/results/minimal_manhattan_realtime_v2_sc_1 60
```
