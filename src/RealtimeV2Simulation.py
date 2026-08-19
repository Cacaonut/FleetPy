# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import datetime
import logging
import os
import queue
import sys
import threading
import time

# additional module imports (> requirements)
# ------------------------------------------
import numpy as np
import pandas as pd
from tqdm import tqdm

# src imports
# -----------
from src.FleetSimulationBase import PROGRESS_LOOP, PROGRESS_LOOP_VEHICLE_STATUS
from src.RealtimeSimulation import RealtimeSimulation, RealtimeMetrics
from src.misc.globals import *

LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_RealtimeV2Simulation = {
    "doc": """
    Real-time simulation environment with decoupled fleet simulation and fleet control threads.
    1. Demand feeder thread injects requests at wall-clock pace into a queue.
    2. Fleet simulation thread runs at wall-clock pace, drains the queue (with Stage 1 timeout), updates vehicle kinematics, and applies optimization results.
    3. Fleet control thread runs optimization (batch ILP assignment, pricing, repositioning, charging strategy) unpaced in the background.
    """,
    "inherit": "RealtimeSimulation",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [
        G_RT_SPEED_FACTOR, G_RT_REQUEST_TIMEOUT, G_RA_REOPT_TS
    ],
    "mandatory_modules": [],
    "optional_modules": []
}


# -------------------------------------------------------------------------------------------------------------------- #
# helper metrics class
# --------------------
class RealtimeV2Metrics(RealtimeMetrics):
    """Extends RealtimeMetrics with dedicated tracking for asynchronous optimization cycles."""

    def __init__(self):
        super().__init__()
        self.optimization_records = []

    def record_optimization(self, sim_time, duration_s, speed_factor=1.0, reopt_budget=None):
        unscaled_duration = duration_s
        scaled_duration = duration_s * speed_factor
        is_lag = scaled_duration > reopt_budget if reopt_budget is not None and reopt_budget > 0 else False
        with self._lock:
            self.optimization_records.append({
                "sim_time": sim_time,
                "unscaled_optimization_duration_s": unscaled_duration,
                "scaled_optimization_duration_s": scaled_duration,
                "optimization_duration_s": scaled_duration,
                "reopt_budget_s": reopt_budget,
                "is_lag": is_lag
            })
        if is_lag:
            LOG.warning(f"Optimization started at sim_time={sim_time} took {scaled_duration:.2f}s (sim) / "
                        f"{unscaled_duration:.3f}s (wall) (reopt budget: {reopt_budget}s sim) — lag detected")

    def save(self, output_dir):
        """Write request, tick, optimization, and summary metrics to CSV files."""
        super().save(output_dir)

        # per-optimization metrics
        if self.optimization_records:
            opt_df = pd.DataFrame(self.optimization_records)
            opt_df.to_csv(os.path.join(output_dir, "rt_optimization_metrics.csv"), index=False)

        # update summary with optimization stats if summary file exists
        summary_f = os.path.join(output_dir, "rt_summary.csv")
        if os.path.isfile(summary_f) and self.optimization_records:
            try:
                summary_df = pd.read_csv(summary_f)
                scaled_durations = [o["scaled_optimization_duration_s"] for o in self.optimization_records]
                unscaled_durations = [o["unscaled_optimization_duration_s"] for o in self.optimization_records]
                summary_df["total_optimizations"] = len(self.optimization_records)
                summary_df["avg_scaled_opt_duration_s"] = float(np.mean(scaled_durations))
                summary_df["max_scaled_opt_duration_s"] = float(np.max(scaled_durations))
                summary_df["p95_scaled_opt_duration_s"] = float(np.percentile(scaled_durations, 95))
                summary_df["avg_unscaled_opt_duration_s"] = float(np.mean(unscaled_durations))
                summary_df["max_unscaled_opt_duration_s"] = float(np.max(unscaled_durations))
                summary_df["p95_unscaled_opt_duration_s"] = float(np.percentile(unscaled_durations, 95))
                summary_df["opt_lag_count"] = sum(1 for o in self.optimization_records if o.get("is_lag"))
                summary_df.to_csv(summary_f, index=False)
            except Exception as e:
                LOG.warning(f"Could not update rt_summary.csv with optimization stats: {e}")

    def print_summary(self):
        """Format and return a readable string of key metrics including optimization stats."""
        base_summary = super().print_summary()
        if not self.optimization_records:
            return base_summary

        scaled_opt = [o["scaled_optimization_duration_s"] for o in self.optimization_records]
        unscaled_opt = [o["unscaled_optimization_duration_s"] for o in self.optimization_records]
        total_opts = len(self.optimization_records)
        lag_opts = sum(1 for o in self.optimization_records if o.get("is_lag"))
        pct_lag = (lag_opts / total_opts * 100) if total_opts else 0.0

        col_w = 44
        lines = base_summary.split("\n")
        # insert before the last divider
        opt_lines = [
            f"  {'Total Optimization Cycles':<{col_w}}: {total_opts} (Lags: {lag_opts} [{pct_lag:.1f}%])",
            f"  {'Optimization Duration (s)':<{col_w}}: avg={np.mean(scaled_opt):.2f}s, max={np.max(scaled_opt):.2f}s, "
            f"p95={np.percentile(scaled_opt, 95):.2f}s (unscaled: avg={np.mean(unscaled_opt):.3f}s, max={np.max(unscaled_opt):.3f}s)"
        ]
        if len(lines) >= 2:
            return "\n".join(lines[:-1] + opt_lines + [lines[-1]])
        else:
            return base_summary + "\n" + "\n".join(opt_lines)


# -------------------------------------------------------------------------------------------------------------------- #
# main simulation class
# ----------------------
class RealtimeV2Simulation(RealtimeSimulation):
    """Real-time simulation with decoupled fleet simulation and fleet control.

    - Demand Feeder Thread: puts requests into queue at wall-clock pace.
    - Fleet Simulation Thread: ticks vehicle movements, network, queue drain (Stage 1 timeout), and applies offers/plans.
    - Fleet Control Thread: runs batch optimization, pricing, repositioning, and charging strategy in the background.
    """

    def __init__(self, scenario_parameters):
        super().__init__(scenario_parameters)
        self._metrics = RealtimeV2Metrics()

        # Thread synchronization
        self._state_lock = threading.RLock()
        self._fc_trigger = threading.Event()
        self._fc_done = threading.Event()
        self._fc_running = False
        self._fc_sim_time = self.start_time
        self._pending_stage2_rids = set()

        # Reoptimization interval
        self._reopt_time_step = scenario_parameters.get(G_RA_REOPT_TS, self.time_step)

    # ----- fleet control worker thread -----
    def _fleet_control_worker(self):
        """Worker function for the fleet control thread. Runs optimization whenever triggered."""
        LOG.info("Fleet control thread started.")
        while not self._stop_event.is_set():
            self._fc_trigger.wait(timeout=0.1)
            if not self._fc_trigger.is_set():
               continue
            self._fc_trigger.clear()
            if self._stop_event.is_set():
                break

            sim_time = self._fc_sim_time
            t0 = time.perf_counter()
            LOG.debug(f"Fleet control thread: starting optimization for sim_time={sim_time}")

            success = True
            try:
                for op in self.operators:
                    if not hasattr(op, "_sim_state_lock"):
                        op._sim_state_lock = self._state_lock
                    with self._state_lock:
                        op.time_trigger(sim_time)
            except Exception as e:
                LOG.error(f"Error during fleet control optimization at sim_time={sim_time}: {e}", exc_info=True)
                success = False
                self._fc_running = False

            if success:
                opt_duration = time.perf_counter() - t0
                reopt_budget = self.scenario_parameters.get(G_RA_REOPT_TS, self.time_step)
                self._metrics.record_optimization(sim_time, opt_duration, self._speed_factor, reopt_budget)
                LOG.debug(f"Fleet control thread: finished optimization for sim_time={sim_time} in {opt_duration:.4f}s")
                self._fc_done.set()

        LOG.info("Fleet control thread finished.")

    # ----- fleet simulation thread -----
    def _fleet_simulation_thread(self, tqdm_position=0):
        """Run fleet simulation step() at wall-clock pace with time-based progress bar."""
        wall_start = time.perf_counter()
        sim_start = self.start_time

        sim_times = range(self.start_time, self.end_time, self.time_step)
        pbar = tqdm(sim_times, position=tqdm_position, desc=self.scenario_parameters.get(G_SCENARIO_NAME)) if PROGRESS_LOOP != "off" else sim_times

        for sim_time in pbar:
            if self._stop_event.is_set():
                break

            # sleep until wall-clock time matches this sim_time
            target_wall = wall_start + (sim_time - sim_start) / self._speed_factor
            now = time.perf_counter()
            if target_wall > now:
                self._stop_event.wait(timeout=target_wall - now)
                if self._stop_event.is_set():
                    break

            self.step(sim_time)
            if PROGRESS_LOOP != "off":
                vehicle_counts = self.count_fleet_status()
                info_dict = {"simulation_time": sim_time,
                             "driving": sum([vehicle_counts[x] for x in G_DRIVING_STATUS])}
                info_dict.update({x.display_name: vehicle_counts[x] for x in PROGRESS_LOOP_VEHICLE_STATUS})
                pbar.set_postfix(info_dict)
            self._update_realtime_plots_dict(sim_time)

        LOG.info("Fleet simulation thread finished.")

    # ----- step -----
    def step(self, sim_time):
        """Queue-aware batch step for real-time processing with asynchronous fleet control.

        1) Update fleets and network (physical movement)
        2) Drain queue & apply Stage 1 timeout (drop if waiting too long in queue)
        3) Inform broker of valid new requests (enter Stage 2)
        4) Consume fleet control results if ready (apply plans, process offers)
        5) Trigger fleet control thread if needed (at reopt intervals)
        6) Handle waiting request cancellations
        7) Trigger charging infrastructure
        8) Record stats + tick metrics
        """
        with self._state_lock:
            tick_start = time.perf_counter()

            # 1) fleet & network update
            self.update_sim_state_fleets(sim_time - self.time_step, sim_time)
            new_travel_times = self.routing_engine.update_network(sim_time)
            if new_travel_times:
                self.broker.inform_network_travel_time_update(sim_time)

            # 2) drain queue and process pending requests (Stage 1)
            all_incoming = self._pending_requests
            self._pending_requests = []
            while True:
                try:
                    all_incoming.append(self._request_queue.get_nowait())
                except queue.Empty:
                    break

            valid_requests = []
            for item in all_incoming:
                rid, rq_obj, enqueue_sim_time, enqueue_wall_time = item
                if enqueue_sim_time > sim_time:
                    # Request is scheduled for a future simulation time step
                    self._pending_requests.append(item)
                    continue

                now_wall = time.perf_counter()
                wall_wait_time = now_wall - enqueue_wall_time
                sim_wait_time = wall_wait_time * self._speed_factor

                if sim_wait_time > self._request_timeout:
                    # Stage 1 Timeout: request waited too long in queue before optimization pickup
                    self._request_enqueue_times.pop(rid, None)
                    self._metrics.record_timeout(rid, enqueue_wall_time, now_wall, self._speed_factor)
                    LOG.warning(f"Request {rid} timed out in queue after {sim_wait_time:.2f}s (sim) / {wall_wait_time:.4f}s (wall) "
                                f"(threshold: {self._request_timeout}s)")
                    rq_obj.set_timed_out(sim_time)
                    self.demand.rq_db[rid] = rq_obj
                    self.demand.record_user(rid)
                    self.demand.rq_db.pop(rid, None)
                    self._pending_stage2_rids.discard(rid)
                else:
                    # Valid request: enters Stage 2 (in optimization)
                    self.demand.rq_db[rid] = rq_obj
                    self.demand.undecided_rq[rid] = rq_obj
                    valid_requests.append((rid, rq_obj, enqueue_wall_time))
                    self._pending_stage2_rids.add(rid)

            # 3) inform broker of valid new requests immediately
            for rid, rq_obj, enqueue_wall_time in valid_requests:
                self.broker.inform_request(rid, rq_obj, sim_time)

            # 4) consume fleet control results if optimization finished
            self._consume_fc_results(sim_time)

            # 5) check if fleet control thread should be triggered
            self._maybe_trigger_fc(sim_time)

            # 6) cancellations of already-booked waiting requests
            self._check_waiting_request_cancellations(sim_time)

            # 7) charging infrastructure (physical stations)
            for ch_op_dict in self.charging_operator_dict.values():
                for ch_op in ch_op_dict.values():
                    ch_op.time_trigger(sim_time)

            # 8) record stats + tick metrics
            self.record_stats()
            tick_duration = time.perf_counter() - tick_start
            reopt_budget = self.scenario_parameters.get(G_RA_REOPT_TS, self.time_step)
            self._metrics.record_tick(sim_time, tick_duration, len(valid_requests), self._speed_factor, self.time_step, reopt_budget)

    def _user_leaves_system(self, rid, sim_time):
        """Safely handle user leaving the system without raising KeyError."""
        self.broker.inform_user_leaving_system(rid, sim_time)
        self.demand.record_user(rid)
        self.demand.rq_db.pop(rid, None)
        self.demand.undecided_rq.pop(rid, None)

    # ----- consume optimization results -----
    def _consume_fc_results(self, sim_time):
        """If fleet control optimization has completed, collect offers and process user decisions."""
        if not self._fc_done.is_set():
            return

        self._fc_running = False
        for rid, rq_obj in self.demand.get_undecided_travelers(sim_time):
            amod_offers = self.broker.collect_offers(rid)
            if amod_offers:
                # Ensure object is present in rq_db for potential decision/leave logging
                self.demand.rq_db[rid] = rq_obj
                for op_id, amod_offer in amod_offers.items():
                    rq_obj.receive_offer(op_id, amod_offer, sim_time)
                self._rid_chooses_offer(rid, rq_obj, sim_time)
                self._pending_stage2_rids.discard(rid)
                enqueue_wall_time = self._request_enqueue_times.pop(rid, None)
                if enqueue_wall_time is not None:
                    self._metrics.record_response_start(rid, enqueue_wall_time, time.perf_counter(), self._speed_factor)

        self._fc_done.clear()

    # ----- trigger optimization -----
    def _maybe_trigger_fc(self, sim_time):
        """Trigger the fleet control thread at reoptimization intervals."""
        if self._fc_running:
            if sim_time % self._reopt_time_step == 0:
                LOG.warning(f"Fleet control optimization at sim_time={sim_time} skipped: previous optimization still running.")
            return

        if sim_time % self._reopt_time_step == 0:
            self._fc_sim_time = sim_time
            self._fc_running = True
            self._fc_done.clear()
            self._fc_trigger.set()

    # ----- run -----
    def run(self, tqdm_position=0):
        """Launch demand feeder, fleet simulation, and fleet control on 3 separate threads."""
        self._start_realtime_plot()
        t_run_start = time.perf_counter()

        if not self._started:
            self._started = True
            LOG.info(f"Starting RealtimeV2Simulation with speed_factor={self._speed_factor}, "
                     f"request_timeout={self._request_timeout}s, reopt_time_step={self._reopt_time_step}s")

            demand_thread = threading.Thread(
                target=self._demand_thread, name="rt-demand-feeder", daemon=True
            )
            fleet_sim_thread = threading.Thread(
                target=self._fleet_simulation_thread, args=(tqdm_position,), name="rt-fleet-simulation", daemon=True
            )
            fc_thread = threading.Thread(
                target=self._fleet_control_worker, name="rt-fleet-control", daemon=True
            )

            demand_thread.start()
            fc_thread.start()
            fleet_sim_thread.start()

            try:
                while fleet_sim_thread.is_alive():
                    fleet_sim_thread.join(timeout=0.2)
            except KeyboardInterrupt:
                print("\nReceived KeyboardInterrupt (Ctrl+C). Gracefully shutting down real-time V2 simulation...")
                LOG.warning("KeyboardInterrupt (Ctrl+C) received. Stopping simulation threads...")
                self._stop_event.set()
                self._fc_trigger.set()
                fleet_sim_thread.join(timeout=2.0)
                demand_thread.join(timeout=2.0)
                fc_thread.join(timeout=2.0)
                self.record_stats()
                self.save_final_state()
                if not self.skip_output:
                    self._metrics.save(self.dir_names[G_DIR_OUTPUT])
                    metrics_str = self._metrics.print_summary()
                    print(metrics_str)
                    LOG.info(metrics_str)
                self._end_realtime_plot()
                sys.exit(0)

            self._stop_event.set()
            self._fc_trigger.set()
            demand_thread.join(timeout=5.0)
            fc_thread.join(timeout=5.0)

            # finalize
            self.record_stats()
            self.save_final_state()
            self.record_remaining_assignments()
            self.demand.record_remaining_users()
            if not self.skip_output:
                self._metrics.save(self.dir_names[G_DIR_OUTPUT])

        if self.skip_output:
            return

        t_run_end = time.perf_counter()
        self.evaluate()
        t_eval_end = time.perf_counter()

        t_init = datetime.timedelta(seconds=int(t_run_start - self.t_init_start))
        t_sim = datetime.timedelta(seconds=int(t_run_end - t_run_start))
        t_eval = datetime.timedelta(seconds=int(t_eval_end - t_run_end))
        prt_str = f"Scenario {self.scenario_name} finished:\n" \
                  f"{'initialization':>20} : {t_init} h\n" \
                  f"{'simulation':>20} : {t_sim} h\n" \
                  f"{'evaluation':>20} : {t_eval} h\n"
        metrics_str = self._metrics.print_summary()
        full_output = f"{prt_str}\n{metrics_str}\n"
        print(full_output)
        LOG.info(full_output)
        self._end_realtime_plot()
