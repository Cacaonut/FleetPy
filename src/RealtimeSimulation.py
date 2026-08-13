# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging
import os
import queue
import sys
import threading
import time

# additional module imports (> requirements)
# ------------------------------------------
import pandas as pd
import numpy as np
from tqdm import tqdm

# src imports
# -----------
from src.FleetSimulationBase import FleetSimulationBase, PROGRESS_LOOP, PROGRESS_LOOP_VEHICLE_STATUS

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *
LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_RealtimeSimulation = {
    "doc": """
    Real-time simulation environment where fleet control and demand injection run on separate threads.
    Demand is fed via a thread-safe queue; fleet control drains the queue in batches at each tick.
    Time progresses at wall-clock pace, scaled by a configurable speed factor.
    """,
    "inherit": "FleetSimulationBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [
        G_RT_SPEED_FACTOR, G_RT_REQUEST_TIMEOUT
    ],
    "mandatory_modules": [],
    "optional_modules": []
}


# -------------------------------------------------------------------------------------------------------------------- #
# helper class
# ------------
class RealtimeMetrics:
    """Collects real-time simulation performance metrics."""

    def __init__(self):
        self.request_records = []   # per-request metrics
        self.tick_records = []      # per-tick metrics
        self.timeout_count = 0
        self._lock = threading.Lock()

    def record_response_start(self, rid, wall_enqueue_time, wall_process_time, speed_factor=1.0):
        unscaled_response_s = wall_process_time - wall_enqueue_time
        scaled_response_s = unscaled_response_s * speed_factor
        with self._lock:
            self.request_records.append({
                "rid": rid,
                "wall_enqueue_time": wall_enqueue_time,
                "wall_process_time": wall_process_time,
                "scaled_response_time_s": scaled_response_s,
                "unscaled_response_time_s": unscaled_response_s,
                "response_time_s": scaled_response_s,
                "status": "processed"
            })

    def record_timeout(self, rid, wall_enqueue_time, wall_timeout_time, speed_factor=1.0):
        unscaled_response_s = wall_timeout_time - wall_enqueue_time
        scaled_response_s = unscaled_response_s * speed_factor
        with self._lock:
            self.request_records.append({
                "rid": rid,
                "wall_enqueue_time": wall_enqueue_time,
                "wall_process_time": wall_timeout_time,
                "scaled_response_time_s": scaled_response_s,
                "unscaled_response_time_s": unscaled_response_s,
                "response_time_s": scaled_response_s,
                "status": "timed_out"
            })
            self.timeout_count += 1

    def record_tick(self, sim_time, duration_s, n_requests):
        with self._lock:
            self.tick_records.append({
                "sim_time": sim_time,
                "tick_duration_s": duration_s,
                "requests_processed": n_requests,
            })
        if duration_s > 1.0:
            LOG.warning(f"Tick at sim_time={sim_time} took {duration_s:.3f}s (>1s) "
                        f"— real-time lag detected")

    def save(self, output_dir):
        """Write metrics to CSV files in the output directory."""
        # per-request metrics
        if self.request_records:
            rq_df = pd.DataFrame(self.request_records)
            rq_df.to_csv(os.path.join(output_dir, "rt_request_metrics.csv"), index=False)

        # per-tick metrics
        if self.tick_records:
            tick_df = pd.DataFrame(self.tick_records)
            tick_df.to_csv(os.path.join(output_dir, "rt_tick_metrics.csv"), index=False)

        # summary
        if self.request_records:
            scaled_response_times = [r["scaled_response_time_s"] for r in self.request_records
                                     if r["status"] == "processed"]
            unscaled_response_times = [r["unscaled_response_time_s"] for r in self.request_records
                                       if r["status"] == "processed"]
            summary = {
                "total_requests": len(self.request_records),
                "processed": len(scaled_response_times),
                "timed_out": self.timeout_count,
                # scaled metrics (simulated time = unscaled * speed_factor)
                "avg_scaled_response_time_s": np.mean(scaled_response_times) if scaled_response_times else 0,
                "max_scaled_response_time_s": np.max(scaled_response_times) if scaled_response_times else 0,
                "p95_scaled_response_time_s": np.percentile(scaled_response_times, 95) if scaled_response_times else 0,
                "p99_scaled_response_time_s": np.percentile(scaled_response_times, 99) if scaled_response_times else 0,
                # unscaled metrics (actual wall-clock time)
                "avg_unscaled_response_time_s": np.mean(unscaled_response_times) if unscaled_response_times else 0,
                "max_unscaled_response_time_s": np.max(unscaled_response_times) if unscaled_response_times else 0,
                "p95_unscaled_response_time_s": np.percentile(unscaled_response_times, 95) if unscaled_response_times else 0,
                "p99_unscaled_response_time_s": np.percentile(unscaled_response_times, 99) if unscaled_response_times else 0,
                # legacy aliases for backwards compatibility
                "avg_response_time_s": np.mean(scaled_response_times) if scaled_response_times else 0,
                "max_response_time_s": np.max(scaled_response_times) if scaled_response_times else 0,
            }
            if self.tick_records:
                tick_durations = [t["tick_duration_s"] for t in self.tick_records]
                summary["avg_tick_duration_s"] = np.mean(tick_durations)
                summary["max_tick_duration_s"] = np.max(tick_durations)
                summary["ticks_over_1s"] = sum(1 for d in tick_durations if d > 1.0)
            summary_df = pd.DataFrame([summary])
            summary_df.to_csv(os.path.join(output_dir, "rt_summary.csv"), index=False)
            LOG.info(f"Realtime metrics summary: {summary}")


# -------------------------------------------------------------------------------------------------------------------- #
# main
# ----
class RealtimeSimulation(FleetSimulationBase):
    """Real-time simulation where fleet control and demand run on separate threads.

    The demand thread reads pre-loaded requests from the CSV and injects them into a
    thread-safe queue at wall-clock pace (scaled by speed_factor). The fleet control
    thread drains this queue in batches at each simulation tick.
    """

    def __init__(self, scenario_parameters):
        super().__init__(scenario_parameters)
        self._request_queue = queue.Queue()
        self._speed_factor = scenario_parameters.get(G_RT_SPEED_FACTOR, 1.0)
        self._request_timeout = scenario_parameters.get(G_RT_REQUEST_TIMEOUT, 30)
        self._stop_event = threading.Event()
        self._request_enqueue_times = {}  # rid -> wall_enqueue_time
        self._pending_requests = []       # requests received early relative to sim_time
        self._metrics = RealtimeMetrics()

    def check_sim_env_spec_inputs(self, scenario_parameters):
        pass

    def add_init(self, scenario_parameters):
        super().add_init(scenario_parameters)

    def add_evaluate(self):
        pass

    def _update_realtime_plots_dict(self, sim_time):
        super()._update_realtime_plots_dict(sim_time)
        if self._shared_dict is not None:
            self._shared_dict["is_realtime"] = True

    # ----- demand thread -----
    def _demand_thread(self):
        """Inject requests from pre-loaded demand at wall-clock pace."""
        sorted_times = sorted(self.demand.future_requests.keys())
        if not sorted_times:
            LOG.info("Demand thread: no future requests to inject.")
            return

        wall_start = time.perf_counter()
        sim_start = self.start_time

        for rq_time in sorted_times:
            if self._stop_event.is_set():
                break
            if rq_time >= self.end_time:
                break

            # sleep until wall-clock time matches this request time
            target_wall = wall_start + (rq_time - sim_start) / self._speed_factor
            now = time.perf_counter()
            if target_wall > now:
                self._stop_event.wait(timeout=target_wall - now)
                if self._stop_event.is_set():
                    break

            # pop requests for this time and enqueue them
            rqs = self.demand.future_requests.pop(rq_time, {})
            for rid, rq_obj in rqs.items():
                rq_obj.set_direct_route_travel_infos(self.routing_engine)
                now_wall = time.perf_counter()
                self._request_enqueue_times[rid] = now_wall
                self._request_queue.put((rid, rq_obj, rq_time, now_wall))

        LOG.info("Demand thread finished.")

    # ----- fleet control thread -----
    def _fleet_control_thread(self, tqdm_position=0):
        """Run fleet control step() at wall-clock pace with time-based progress bar."""
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

        LOG.info("Fleet control thread finished.")

    # ----- step -----
    def step(self, sim_time):
        """Queue-aware batch step for real-time processing.

        1) update fleets and network
        2) drain queue and separate valid requests from timed-out / future requests
        3) inform broker of valid new requests
        4) operator batch optimization
        5) collect offers + user decisions for undecided travelers
        6) handle waiting request cancellations
        7) trigger charging infrastructure
        8) record stats + tick metrics

        :param sim_time: new simulation time
        :return: None
        """
        tick_start = time.perf_counter()

        # 1) fleet & network update
        self.update_sim_state_fleets(sim_time - self.time_step, sim_time)
        new_travel_times = self.routing_engine.update_network(sim_time)
        if new_travel_times:
            self.broker.inform_network_travel_time_update(sim_time)

        # 2) drain queue and process pending requests
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
                # Request timed out in queue before processing
                self._request_enqueue_times.pop(rid, None)
                self._metrics.record_timeout(rid, enqueue_wall_time, now_wall, self._speed_factor)
                LOG.warning(f"Request {rid} timed out after {sim_wait_time:.2f}s (sim) / {wall_wait_time:.4f}s (wall) "
                            f"(threshold: {self._request_timeout}s)")
                rq_obj.set_timed_out(sim_time)
                self.demand.rq_db[rid] = rq_obj
                self.demand.record_user(rid)
                self.demand.rq_db.pop(rid, None)
            else:
                self.demand.rq_db[rid] = rq_obj
                self.demand.undecided_rq[rid] = rq_obj
                valid_requests.append((rid, rq_obj, enqueue_wall_time))

        # 3) inform broker of valid new requests
        for rid, rq_obj, enqueue_wall_time in valid_requests:
            self.broker.inform_request(rid, rq_obj, sim_time)

        # 4) operator batch optimization
        for op in self.operators:
            op.time_trigger(sim_time)

        # 5) collect offers + decisions
        for rid, rq_obj in self.demand.get_undecided_travelers(sim_time):
            amod_offers = self.broker.collect_offers(rid)
            for op_id, amod_offer in amod_offers.items():
                rq_obj.receive_offer(op_id, amod_offer, sim_time)
            self._rid_chooses_offer(rid, rq_obj, sim_time)
            enqueue_wall_time = self._request_enqueue_times.pop(rid, None)
            if enqueue_wall_time is not None:
                self._metrics.record_response_start(rid, enqueue_wall_time, time.perf_counter(), self._speed_factor)

        # 6) cancellations
        self._check_waiting_request_cancellations(sim_time)

        # 7) charging
        for ch_op_dict in self.charging_operator_dict.values():
            for ch_op in ch_op_dict.values():
                ch_op.time_trigger(sim_time)

        # 8) record stats + tick metrics
        self.record_stats()
        tick_duration = time.perf_counter() - tick_start
        self._metrics.record_tick(sim_time, tick_duration, len(valid_requests))

    # ----- run -----
    def run(self, tqdm_position=0):
        """Launch demand and fleet control on separate threads."""
        import datetime
        self._start_realtime_plot()
        t_run_start = time.perf_counter()

        if not self._started:
            self._started = True
            LOG.info(f"Starting RealtimeSimulation with speed_factor={self._speed_factor}, "
                     f"request_timeout={self._request_timeout}s")

            demand_thread = threading.Thread(
                target=self._demand_thread, name="rt-demand-feeder", daemon=True
            )
            fleet_thread = threading.Thread(
                target=self._fleet_control_thread, args=(tqdm_position,), name="rt-fleet-control", daemon=True
            )

            demand_thread.start()
            fleet_thread.start()

            try:
                while fleet_thread.is_alive():
                    fleet_thread.join(timeout=0.2)
            except KeyboardInterrupt:
                print("\nReceived KeyboardInterrupt (Ctrl+C). Gracefully shutting down real-time simulation...")
                LOG.warning("KeyboardInterrupt (Ctrl+C) received. Stopping real-time simulation threads...")
                self._stop_event.set()
                fleet_thread.join(timeout=2.0)
                demand_thread.join(timeout=2.0)
                self.record_stats()
                self.save_final_state()
                if not self.skip_output:
                    self._metrics.save(self.dir_names[G_DIR_OUTPUT])
                self._end_realtime_plot()
                sys.exit(0)

            self._stop_event.set()
            demand_thread.join(timeout=5)

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
        print(prt_str)
        LOG.info(prt_str)
        self._end_realtime_plot()
