# Minimal Manhattan Realtime Ride-Pooling Study

This study defines a self-contained real-time ride-pooling simulation using the Manhattan benchmark dataset in FleetPy.

## 📂 Study Directory Layout

```
studies/minimal_manhattan_realtime/
├── README.md                           # Study documentation & workflow guide
├── download_data.py                    # Automated dataset downloader & file organizer
├── run_minimal_manhattan_realtime.py   # Real-time simulation execution script
└── scenarios/
    ├── constant_config.yaml            # Environment, network, & realtime parameters (`sim_env: RealtimeSimulation`)
    └── scenario_config.csv             # Scenario definition (`minimal_manhattan_realtime_sc_1`)
```

---

## 📥 Automated Dataset Download & Setup

This study uses the same Manhattan dataset from Zenodo:

```bash
python studies/minimal_manhattan_realtime/download_data.py
```

---

## 🚀 Execution & Results

### 1. Run Realtime Simulation:
```bash
python studies/minimal_manhattan_realtime/run_minimal_manhattan_realtime.py
```

Real-time simulation results are written to `studies/minimal_manhattan_realtime/results/minimal_manhattan_realtime_sc_1/`:
- `standard_eval.csv` (Aggregated KPIs: wait times, modal split, vehicle miles, pooled trip share)
- `1_user-stats.csv` (Per-trip passenger stats)
- `2-0_op-stats.csv` (Operator vehicle task logs)
- `rt_request_metrics.csv` (Per-request enqueue, process, and response times; processed vs timed_out status)
- `rt_tick_metrics.csv` (Per-tick duration and request throughput)
- `rt_summary.csv` (Aggregated response time & tick performance metrics)

### 2. Replay Visualization:
To visually replay vehicle movements and operational metrics:
```bash
python replay_pyplot.py studies/minimal_manhattan_realtime/results/minimal_manhattan_realtime_sc_1 60
```
