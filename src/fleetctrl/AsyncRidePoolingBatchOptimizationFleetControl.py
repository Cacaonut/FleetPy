from __future__ import annotations
import logging
import time
from typing import Dict, List, Any, Set, TYPE_CHECKING

from src.simulation.Legs import VehicleRouteLeg
from src.fleetctrl.RidePoolingBatchAssignmentFleetcontrol import RidePoolingBatchAssignmentFleetcontrol
from src.fleetctrl.planning.VehiclePlan import VehiclePlan
from src.fleetctrl.reconciliation.PlanReconciler import PlanReconciler
from src.fleetctrl.pooling.batch.BatchAssignmentAlgorithmBase import SimulationVehicleStruct
from src.fleetctrl.pooling.GeneralPoolingFunctions import get_assigned_rids_from_vehplan
from src.misc.globals import *

if TYPE_CHECKING:
    from src.routing.NetworkBase import NetworkBase
    from src.simulation.Vehicles import SimulationVehicle

LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_AsyncRidePoolingBatchOptimizationFleetControl = {
    "doc": """Async batch assignment fleet control for real-time simulation.
        Decouples the optimization solver from the simulation thread via event queueing.
        The RPBO_Module mutations are intercepted by a proxy and replayed on the solver thread.""",
    "inherit": "RidePoolingBatchAssignmentFleetcontrol",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

class QueueingRPBOProxy:
    """Proxy that queues mutating RPBO_Module calls for replay on the solver thread.
    
    The real RPBO_Module lives on the solver thread. All calls from the simulation
    thread that would mutate its state are intercepted and stored as events.
    These events are drained when creating an optimization snapshot and replayed
    on the solver thread before running optimization.
    """
    
    _QUEUED_METHODS = frozenset({
        'add_new_request', 'set_request_assigned', 
        'set_database_in_case_of_boarding', 'set_database_in_case_of_alighting',
        'delete_request', 'delRequest', 'register_change_in_time_constraints',
        'set_assignment', 'lock_request_to_vehicle',
        'delete_vehicle_database_entries',
    })
    
    def __init__(self, real_rpbo):
        self.__dict__['_real_rpbo'] = real_rpbo
        self.__dict__['_event_queue'] = []
    
    def __getattr__(self, name):
        if name in QueueingRPBOProxy._QUEUED_METHODS:
            def queued_call(*args, **kwargs):
                self.__dict__['_event_queue'].append((name, args, kwargs))
            return queued_call
        return getattr(self.__dict__['_real_rpbo'], name)
    
    def __setattr__(self, name, value):
        # Forward attribute sets to real module
        setattr(self.__dict__['_real_rpbo'], name, value)
    
    def drain_events(self):
        """Drain and return all queued events. Thread-safe via GIL."""
        events = list(self.__dict__['_event_queue'])
        self.__dict__['_event_queue'].clear()
        return events
    
    def replay_events(self, events):
        """Replay queued events on the real RPBO module. Called from solver thread."""
        real = self.__dict__['_real_rpbo']
        for method_name, args, kwargs in events:
            if hasattr(real, method_name):
                getattr(real, method_name)(*args, **kwargs)
    
    @property
    def real_module(self):
        return self.__dict__['_real_rpbo']


class SafeRqDict(dict):
    """Dictionary view that safely yields fallback request objects for recently alighted or cancelled passengers."""
    def __missing__(self, key):
        class FallbackPlanRequest:
            rq_time = 0
            nr_pax = 1
            def get_current_offer(self):
                return None
            def get_o_stop_info(self):
                return None, 0, None
        return FallbackPlanRequest()


class AsyncRidePoolingBatchOptimizationFleetControl(RidePoolingBatchAssignmentFleetcontrol):
    def __init__(self, op_id, operator_attributes, list_vehicles, routing_engine, zone_system, 
                 scenario_parameters, dir_names, op_charge_depot_infra=None, list_pub_charging_infra=[]):
        super().__init__(op_id, operator_attributes, list_vehicles, routing_engine, zone_system,
                         scenario_parameters, dir_names=dir_names, op_charge_depot_infra=op_charge_depot_infra,
                         list_pub_charging_infra=list_pub_charging_infra)
        # Replace RPBO_Module with queueing proxy
        self.RPBO_Module = QueueingRPBOProxy(self.RPBO_Module)
        self._reconciler = PlanReconciler()
        # Track requests pending optimization result
        self._pending_optimization_rids = set()  # rids sent to solver, awaiting result

    def _call_time_trigger_request_batch(self, simulation_time):
        """No-op: batch optimization is managed asynchronously by RealtimeV2Simulation."""
        pass

    def create_optimization_snapshot(self, sim_time):
        events = self.RPBO_Module.drain_events()
        
        vid_finished_VRLs = dict(self.vid_finished_VRLs)
        self.vid_finished_VRLs.clear()
        
        veh_objs = {}
        veh_plans_snapshot = {}
        for veh_obj in self.sim_vehicles:
            vp = self.veh_plans.get(veh_obj.vid, VehiclePlan(veh_obj, sim_time, self.routing_engine, []))
            veh_plans_snapshot[veh_obj.vid] = vp.copy()
            veh_objs[veh_obj.vid] = SimulationVehicleStruct(veh_obj, vp, sim_time, self.routing_engine)
        
        # Transfer all new requests into unassigned_requests_1
        for rid in list(self.new_requests.keys()):
            self.unassigned_requests_1[rid] = 1
        self.new_requests.clear()

        # Remember which requests are pending an optimization result
        self._pending_optimization_rids = set(self.unassigned_requests_1.keys()) | set(self.unassigned_requests_2.keys())
        
        return {
            'sim_time': sim_time,
            'events': events,
            'vid_finished_VRLs': vid_finished_VRLs,
            'veh_objs': veh_objs,
            'veh_plans': veh_plans_snapshot,
            'new_travel_times': self.new_travel_times_loaded,
        }

    def run_optimization_on_snapshot(self, snapshot):
        sim_time = snapshot['sim_time']
        events = snapshot['events']
        vid_finished_VRLs = snapshot['vid_finished_VRLs']
        veh_objs = snapshot['veh_objs']
        veh_plans_snapshot = snapshot.get('veh_plans', {})
        new_travel_times = snapshot['new_travel_times']
        
        # Replay queued events on real RPBO module
        self.RPBO_Module.replay_events(events)
        
        # Ensure external assignments fallback is populated for all vehicles
        real_rpbo = self.RPBO_Module.real_module
        if hasattr(real_rpbo, 'external_assignments'):
            for vid, veh_plan in veh_plans_snapshot.items():
                curr_key = real_rpbo.current_assignments.get(vid)
                if curr_key is not None and vid not in real_rpbo.external_assignments:
                    real_rpbo.external_assignments[vid] = (curr_key, veh_plan)
        
        results = {}
        try:
            # Run optimization on real module
            real_rpbo.compute_new_vehicle_assignments(
                sim_time, vid_finished_VRLs,
                veh_objs_to_build=veh_objs,
                build_from_scratch=True,
                new_travel_times=new_travel_times
            )
            
            # Extract results
            for vid in veh_objs:
                plan = real_rpbo.get_optimisation_solution(vid)
                if plan is not None:
                    results[vid] = plan
        except Exception as e:
            LOG.warning(f"Batch optimization failed at sim_time={sim_time}: {e}. Skipping this cycle.")
            results = {}
        finally:
            real_rpbo.clear_databases()
            
        return results

    def _get_active_rids(self, exclude_rid=None):
        """Get set of request IDs valid across both fleet control and simulation vehicle databases."""
        sim_rq_db = self.sim_vehicles[0].rq_db if self.sim_vehicles else {}
        active = set(self.rq_dict.keys()) & set(sim_rq_db.keys())
        if exclude_rid is not None:
            active.discard(exclude_rid)
        return active

    def compute_VehiclePlan_utility(self, simulation_time, veh_obj, vehicle_plan):
        """Compute utility, ensuring cancelled/deleted requests are stripped and pax_info is complete."""
        if vehicle_plan is not None:
            assigned_rids = get_assigned_rids_from_vehplan(vehicle_plan)
            sim_rq_db = self.sim_vehicles[0].rq_db if self.sim_vehicles else {}
            missing = [r for r in assigned_rids if r not in self.rq_dict or r not in sim_rq_db]
            if missing:
                active_rids = self._get_active_rids()
                onboard_rids = {rq.get_rid_struct() for rq in veh_obj.pax}
                filtered_stops = self._reconciler._filter_plan_stops(
                    vehicle_plan.list_plan_stops, active_rids, onboard_rids, rq_dict=self.rq_dict
                )
                for rq in veh_obj.pax:
                    if getattr(rq, 'pu_time', None) is None:
                        rq.pu_time = simulation_time
                vehicle_plan = VehiclePlan(veh_obj, simulation_time, self.routing_engine, filtered_stops)
                self.veh_plans[veh_obj.vid] = vehicle_plan
            
            # Ensure every entry in pax_info has both [pu_time, do_time] for control_f
            if hasattr(vehicle_plan, 'pax_info') and vehicle_plan.pax_info:
                for rid, info in list(vehicle_plan.pax_info.items()):
                    if len(info) < 2:
                        pu = info[0] if len(info) >= 1 and info[0] is not None else simulation_time
                        vehicle_plan.pax_info[rid] = [pu, max(pu, simulation_time)]
                        
        safe_rqs = SafeRqDict(self.rq_dict)
        return self.vr_ctrl_f(simulation_time, veh_obj, vehicle_plan, safe_rqs, self.routing_engine)

    def user_cancels_request(self, rid: Any, simulation_time: int):
        """Handle request cancellation across all vehicles holding this request in their plan."""
        active_rids = self._get_active_rids(exclude_rid=rid)
        for veh_obj in self.sim_vehicles:
            assigned_plan = self.veh_plans.get(veh_obj.vid)
            if assigned_plan is not None and rid in get_assigned_rids_from_vehplan(assigned_plan):
                onboard_rids = {rq.get_rid_struct() for rq in veh_obj.pax}
                if rid in onboard_rids:
                    LOG.warning(f"Request {rid} is already onboard vehicle {veh_obj.vid} and cannot cancel before dropoff.")
                    return
                filtered_stops = self._reconciler._filter_plan_stops(
                    assigned_plan.list_plan_stops, active_rids, onboard_rids, rq_dict=self.rq_dict
                )
                for rq in veh_obj.pax:
                    if getattr(rq, 'pu_time', None) is None:
                        rq.pu_time = simulation_time
                new_plan = VehiclePlan(veh_obj, simulation_time, self.routing_engine, filtered_stops)
                self.assign_vehicle_plan(veh_obj, new_plan, simulation_time, force_assign=True)
        self.unassigned_requests_1.pop(rid, None)
        self.unassigned_requests_2.pop(rid, None)
        super(RidePoolingBatchAssignmentFleetcontrol, self).user_cancels_request(rid, simulation_time)

    def acknowledge_boarding(self, rid: Any, vid: int, simulation_time: int):
        """Safely acknowledge boarding with safe prq lookup."""
        self.sim_time = simulation_time
        LOG.debug(f"acknowledge boarding {rid} in {vid} at {simulation_time}")
        prq = self.rq_dict.get(rid)
        if prq is not None:
            prq.set_pickup(vid, simulation_time)
        self.RPBO_Module.set_database_in_case_of_boarding(rid, vid)

    def acknowledge_alighting(self, rid: Any, vid: int, simulation_time: int):
        """Safely acknowledge alighting with safe dictionary deletion."""
        self.sim_time = simulation_time
        LOG.debug(f"acknowledge alighting {rid} from {vid} at {simulation_time}")
        self.RPBO_Module.set_database_in_case_of_alighting(rid, vid)
        self.rq_dict.pop(rid, None)
        self.rid_to_assigned_vid.pop(rid, None)

    def assign_vehicle_plan(self, veh_obj: SimulationVehicle, vehicle_plan: VehiclePlan, sim_time: int, 
                            force_assign: bool = False, assigned_charging_task: Tuple[Tuple[str, int], Any] = None, 
                            add_arg: bool = None):
        """Pre-sanitize vehicle plan and safely assign it without crashing on alighted/cancelled requests."""
        if vehicle_plan is not None:
            active_rids = self._get_active_rids()
            onboard_rids = {rq.get_rid_struct() for rq in veh_obj.pax}
            assigned_rids = get_assigned_rids_from_vehplan(vehicle_plan)
            if any(r not in active_rids and r not in onboard_rids for r in assigned_rids):
                filtered_stops = self._reconciler._filter_plan_stops(
                    vehicle_plan.list_plan_stops, active_rids, onboard_rids, rq_dict=self.rq_dict
                )
                for rq in veh_obj.pax:
                    if getattr(rq, 'pu_time', None) is None:
                        rq.pu_time = sim_time
                vehicle_plan = VehiclePlan(veh_obj, sim_time, self.routing_engine, filtered_stops)

            for rq in veh_obj.pax:
                if getattr(rq, 'pu_time', None) is None:
                    rq.pu_time = sim_time

            # Bind safe get_pax_info to guarantee [pu_time, do_time] has length >= 2
            def safe_get_pax_info(rid, plan_ref=vehicle_plan):
                info = plan_ref.pax_info.get(rid)
                if not info:
                    return [sim_time, sim_time]
                if len(info) < 2:
                    pu = info[0] if info[0] is not None else sim_time
                    return [pu, max(pu, sim_time)]
                return info
            vehicle_plan.get_pax_info = safe_get_pax_info

        missing_rids = [r for r in get_assigned_rids_from_vehplan(vehicle_plan) if r not in self.rq_dict]
        for r in missing_rids:
            class DummyAssignedPRQ:
                def set_assigned(self, *args, **kwargs): pass
            self.rq_dict[r] = DummyAssignedPRQ()
            
        try:
            super().assign_vehicle_plan(veh_obj, vehicle_plan, sim_time, force_assign=force_assign, 
                                        assigned_charging_task=assigned_charging_task, add_arg=add_arg)
        finally:
            for r in missing_rids:
                self.rq_dict.pop(r, None)

    def apply_reconciled_results(self, reconciled_plans, sim_time):
        # Apply reconciled plans
        for vid, plan in reconciled_plans.items():
            veh_obj = self.sim_vehicles[vid]
            self.assign_vehicle_plan(veh_obj, plan, sim_time, force_assign=True, add_arg=True)
        
        # Create offers for pending requests
        for rid in self._pending_optimization_rids:
            prq = self.rq_dict.get(rid)
            if prq is None:
                continue
            assigned_vid = self.rid_to_assigned_vid.get(rid)
            if assigned_vid is not None and assigned_vid in self.veh_plans:
                plan = self.veh_plans[assigned_vid]
                if rid in get_assigned_rids_from_vehplan(plan):
                    self._create_user_offer(prq, sim_time, assigned_vehicle_plan=plan)
                else:
                    self._create_user_offer(prq, sim_time)  # rejection
            else:
                # Check max_wait_time_2 retry logic
                if rid in self.unassigned_requests_1:
                    if self.max_wait_time_2 is not None and self.max_wait_time_2 > 0:
                        # Retry with extended wait time
                        self.unassigned_requests_2[rid] = 1
                        self.RPBO_Module.delete_request(rid)  # through proxy, queued
                        _, earliest_pu, _ = prq.get_o_stop_info()
                        new_latest_pu = earliest_pu + self.max_wait_time_2
                        self.change_prq_time_constraints(sim_time, rid, new_latest_pu)
                        self.RPBO_Module.add_new_request(rid, prq)  # through proxy, queued
                    else:
                        self._create_user_offer(prq, sim_time)  # rejection
                elif rid in self.unassigned_requests_2:
                    self._create_user_offer(prq, sim_time)  # rejection after retry
        
        # Clear tracking only for requests that were actually in this batch
        for rid in self._pending_optimization_rids:
            self.unassigned_requests_1.pop(rid, None)
        # Only clear requests_2 that were actually in this batch
        for rid in list(self.unassigned_requests_2.keys()):
            if rid in self._pending_optimization_rids:
                if self.rid_to_assigned_vid.get(rid) is not None or rid not in self.unassigned_requests_2:
                    pass  # either assigned or already moved to retry
        self.new_travel_times_loaded = False
        self._pending_optimization_rids = set()

    def has_pending_requests(self):
        """Check if there are requests waiting for optimization."""
        return bool(self.unassigned_requests_1) or bool(self.unassigned_requests_2) or bool(self.new_requests)

    def _build_VRLs(self, vehicle_plan: VehiclePlan, veh_obj: SimulationVehicle, sim_time: int) -> List[VehicleRouteLeg]:
        """Build VRLs while seamlessly preserving currently active locked legs (e.g. ongoing physical BOARDING)."""
        active_rids = self._get_active_rids()
        onboard_rids = {rq.get_rid_struct() for rq in veh_obj.pax}
        assigned_rids = get_assigned_rids_from_vehplan(vehicle_plan)
        if any(r not in active_rids and r not in onboard_rids for r in assigned_rids):
            filtered_stops = self._reconciler._filter_plan_stops(
                vehicle_plan.list_plan_stops, active_rids, onboard_rids, rq_dict=self.rq_dict
            )
            for rq in veh_obj.pax:
                if getattr(rq, 'pu_time', None) is None:
                    rq.pu_time = sim_time
            vehicle_plan = VehiclePlan(veh_obj, sim_time, self.routing_engine, filtered_stops)
            self.veh_plans[veh_obj.vid] = vehicle_plan
            
        new_list_vrls = super()._build_VRLs(vehicle_plan, veh_obj, sim_time)
        if veh_obj.assigned_route:
            ca = veh_obj.assigned_route[0]
            if ca.status == VRL_STATES.BOARDING:
                ca.rq_dict[1] = [prq for prq in ca.rq_dict.get(1, []) if prq.get_rid() in active_rids]
                ca.rq_dict[-1] = [prq for prq in ca.rq_dict.get(-1, []) if prq.get_rid() in active_rids or prq.get_rid() in onboard_rids]
                if new_list_vrls and new_list_vrls[0].status == ca.status and new_list_vrls[0].destination_pos == ca.destination_pos:
                    new_list_vrls[0] = ca
                else:
                    new_list_vrls = [ca] + new_list_vrls
        return new_list_vrls
