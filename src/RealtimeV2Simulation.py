import logging
import queue
import sys
import threading
import time

from src.RealtimeSimulation import RealtimeSimulation
from src.fleetctrl.reconciliation.PlanReconciler import PlanReconciler
from src.misc.globals import *

LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_RealtimeV2Simulation = {
    "doc": """
    Real-time simulation V2 environment. Uses 3 threads:
    - Demand thread: injects requests
    - Fleet control thread: steps vehicle physics, drains queue, consumes results
    - Solver thread: blocks on queue, runs batch optimization
    """,
    "inherit": "RealtimeSimulation",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [
        G_RT_MIN_REOPT_INTERVAL
    ],
    "mandatory_modules": [],
    "optional_modules": []
}


class RealtimeV2Simulation(RealtimeSimulation):
    """
    Real-time simulation V2 where fleet control and optimization are fully asynchronous.
    """

    def __init__(self, scenario_parameters):
        super().__init__(scenario_parameters)
        self._to_solver_queue = queue.Queue(maxsize=1)
        self._from_solver_queue = queue.Queue(maxsize=1)
        self._reconciler = PlanReconciler()
        self._solver_idle = True
        self._last_opt_start_time = -float('inf')
        self._min_reopt_interval = scenario_parameters.get(
            G_RT_MIN_REOPT_INTERVAL,
            scenario_parameters.get(G_RA_REOPT_TS, 30)
        )

    def _solver_thread(self):
        """Solver thread: blocks on queue, runs optimization, returns results."""
        LOG.info("Solver thread started.")
        while not self._stop_event.is_set():
            try:
                snapshot = self._to_solver_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            LOG.info(f"Solver started optimization at sim_time={snapshot['sim_time']}")
            t0 = time.perf_counter()
            
            results = {}
            try:
                for op_id, op in enumerate(self.operators):
                    if hasattr(op, 'run_optimization_on_snapshot'):
                        op_result = op.run_optimization_on_snapshot(snapshot[op_id])
                        results[op_id] = op_result
            except Exception as e:
                LOG.warning(f"Solver thread caught exception at sim_time={snapshot.get('sim_time')}: {e}. Skipping cycle.")
                results = {}
            
            dt = time.perf_counter() - t0
            LOG.info(f"Solver finished optimization in {dt:.3f}s")
            self._from_solver_queue.put(results)
        
        LOG.info("Solver thread finished.")

    def _consume_solver_results(self, sim_time):
        """Check for solver results and reconcile plans."""
        try:
            results = self._from_solver_queue.get_nowait()
        except queue.Empty:
            return
        
        self._solver_idle = True
        LOG.info(f"Consuming solver results at sim_time={sim_time}")
        
        for op_id, op in enumerate(self.operators):
            if op_id not in results:
                continue
            solver_plans = results[op_id]
            
            # Build set of currently active rids (ground truth in simulation demand DB)
            active_rids = set(self.demand.rq_db.keys()) & set(op.rq_dict.keys())
            
            # Reconcile
            reconciled = self._reconciler.reconcile_all(
                op.sim_vehicles, op.veh_plans, solver_plans,
                sim_time, self.routing_engine, op.rq_dict, active_rids
            )
            
            # Apply reconciled plans
            op.apply_reconciled_results(reconciled, sim_time)

    def _maybe_trigger_optimization(self, sim_time):
        """Trigger optimization if conditions are met."""
        if not self._solver_idle:
            return
        if sim_time - self._last_opt_start_time < self._min_reopt_interval:
            return
        
        has_pending = False
        for op in self.operators:
            if hasattr(op, 'has_pending_requests') and op.has_pending_requests():
                has_pending = True
                break
        if not has_pending:
            return
        
        # Create snapshots and dispatch
        snapshots = {'sim_time': sim_time}
        for op_id, op in enumerate(self.operators):
            if hasattr(op, 'create_optimization_snapshot'):
                snapshots[op_id] = op.create_optimization_snapshot(sim_time)
        
        self._to_solver_queue.put(snapshots)
        self._solver_idle = False
        self._last_opt_start_time = sim_time
        LOG.info(f"Dispatched optimization snapshot at sim_time={sim_time}")

    def update_sim_state_fleets(self, last_time, next_time, force_update_plan=False):
        """Update vehicle states, safely guarding demand updates against cancelled/timed-out requests."""
        LOG.debug(f"updating MoD state from {last_time} to {next_time}")
        for opid_vid_tuple, veh_obj in sorted(self.sim_vehicles.items(), key=lambda x: self.vehicle_update_order[x[0]]):
            op_id, vid = opid_vid_tuple
            boarding_requests, alighting_requests, passed_VRL, dict_start_alighting = \
                veh_obj.update_veh_state(last_time, next_time)
            if veh_obj.status == VRL_STATES.CHARGING:
                self.vehicle_update_order[opid_vid_tuple] = 0
            else:
                self.vehicle_update_order[opid_vid_tuple] = 1
            for rid, boarding_time_and_pos in boarding_requests.items():
                boarding_time, boarding_pos = boarding_time_and_pos
                LOG.debug(f"rid {rid} boarding at {boarding_time} at pos {boarding_pos}")
                if rid in self.demand.rq_db:
                    self.demand.record_boarding(rid, vid, op_id, boarding_time, pu_pos=boarding_pos)
                    self.broker.acknowledge_user_boarding(op_id, rid, vid, boarding_time)
            for rid, alighting_start_time_and_pos in dict_start_alighting.items():
                alighting_start_time, alighting_pos = alighting_start_time_and_pos
                LOG.debug(f"rid {rid} deboarding at {alighting_start_time} at pos {alighting_pos}")
                if rid in self.demand.rq_db:
                    self.demand.record_alighting_start(rid, vid, op_id, alighting_start_time, do_pos=alighting_pos)
            for rid, alighting_end_time in alighting_requests.items():
                if rid in self.demand.rq_db:
                    self.demand.user_ends_alighting(rid, vid, op_id, alighting_end_time)
                self.broker.acknowledge_user_alighting(op_id, rid, vid, alighting_end_time)
            if len(boarding_requests) > 0 or len(dict_start_alighting) > 0:
                self.broker.receive_status_update(op_id, vid, next_time, passed_VRL, True)
            else:
                self.broker.receive_status_update(op_id, vid, next_time, passed_VRL, force_update_plan)

    def _is_rid_onboard(self, rid):
        """Check if a request is currently onboard any vehicle."""
        for opid_vid_tuple, veh_obj in self.sim_vehicles.items():
            for rq in veh_obj.pax:
                if getattr(rq, 'rid', None) == rid or getattr(rq, 'sub_rid_struct', None) == rid or getattr(rq, 'get_rid_struct', lambda: None)() == rid:
                    return True
        return False

    def _user_leaves_system(self, rid, sim_time):
        """Handle traveler leaving system while safeguarding in-transit/onboard passengers."""
        if self._is_rid_onboard(rid):
            LOG.warning(f"Traveler {rid} attempted to leave system at sim_time={sim_time} but is currently onboard a vehicle. Deferring cleanup until alighting.")
            self.demand.undecided_rq.pop(rid, None)
            return
        
        self.broker.inform_user_leaving_system(rid, sim_time)
        self.demand.record_user(rid)
        self.demand.rq_db.pop(rid, None)
        self.demand.undecided_rq.pop(rid, None)

    def step(self, sim_time):
        """Queue-aware batch step for real-time processing."""
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
            
        # 4) Check for solver results and reconcile
        self._consume_solver_results(sim_time)

        # 5) Trigger additional tasks (repo, charging strategy, pricing)
        for op in self.operators:
            op.time_trigger(sim_time)

        # 6) Maybe trigger new optimization
        self._maybe_trigger_optimization(sim_time)

        # 7) collect offers + decisions
        for rid, rq_obj in self.demand.get_undecided_travelers(sim_time):
            amod_offers = self.broker.collect_offers(rid)
            for op_id, amod_offer in amod_offers.items():
                rq_obj.receive_offer(op_id, amod_offer, sim_time)
            self._rid_chooses_offer(rid, rq_obj, sim_time)
            enqueue_wall_time = self._request_enqueue_times.pop(rid, None)
            if enqueue_wall_time is not None:
                self._metrics.record_response_start(rid, enqueue_wall_time, time.perf_counter(), self._speed_factor)

        # 8) cancellations
        self._check_waiting_request_cancellations(sim_time)

        # 9) charging
        for ch_op_dict in self.charging_operator_dict.values():
            for ch_op in ch_op_dict.values():
                ch_op.time_trigger(sim_time)

        # 10) record stats + tick metrics
        self.record_stats()
        tick_duration = time.perf_counter() - tick_start
        reopt_budget = self.scenario_parameters.get(G_RA_REOPT_TS, self.time_step)
        self._metrics.record_tick(sim_time, tick_duration, len(valid_requests), self._speed_factor, self.time_step, reopt_budget)

    def run(self, tqdm_position=0):
        """Launch demand, fleet control, and solver on separate threads."""
        import datetime
        self._start_realtime_plot()
        t_run_start = time.perf_counter()

        if not self._started:
            self._started = True
            LOG.info(f"Starting RealtimeV2Simulation with speed_factor={self._speed_factor}, "
                     f"request_timeout={self._request_timeout}s, "
                     f"min_reopt_interval={self._min_reopt_interval}s")

            demand_thread = threading.Thread(
                target=self._demand_thread, name="rt-demand-feeder", daemon=True
            )
            fleet_thread = threading.Thread(
                target=self._fleet_control_thread, args=(tqdm_position,), name="rt-fleet-control", daemon=True
            )
            solver_thread = threading.Thread(
                target=self._solver_thread, name="rt-solver", daemon=True
            )

            demand_thread.start()
            fleet_thread.start()
            solver_thread.start()

            try:
                while fleet_thread.is_alive():
                    fleet_thread.join(timeout=0.2)
            except KeyboardInterrupt:
                print("\nReceived KeyboardInterrupt (Ctrl+C). Gracefully shutting down real-time simulation...")
                LOG.warning("KeyboardInterrupt (Ctrl+C) received. Stopping real-time simulation threads...")
                self._stop_event.set()
                fleet_thread.join(timeout=2.0)
                demand_thread.join(timeout=2.0)
                solver_thread.join(timeout=2.0)
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
            demand_thread.join(timeout=5)
            solver_thread.join(timeout=5)

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
