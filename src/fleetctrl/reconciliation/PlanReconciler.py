from __future__ import annotations
import logging
from typing import Dict, List, Set, Any, Optional, TYPE_CHECKING, Tuple

from src.fleetctrl.planning.VehiclePlan import PlanStop, VehiclePlan
from src.fleetctrl.pooling.GeneralPoolingFunctions import get_assigned_rids_from_vehplan
from src.misc.globals import *

if TYPE_CHECKING:
    from src.routing.NetworkBase import NetworkBase
    from src.simulation.Vehicles import SimulationVehicle
    from src.fleetctrl.planning.PlanRequest import PlanRequest

LOG = logging.getLogger(__name__)

class PlanReconciler:
    """Reconciles solver-computed vehicle plans with live vehicle state.
    
    When the solver computes plans against a snapshot taken at time t_snap,
    vehicles will have moved by the time results arrive at t_curr. This class
    re-anchors the abstract plan intents to the current vehicle positions,
    filters out invalidated requests, and validates feasibility.
    """
    
    def __init__(self):
        pass  # Stateless reconciler
    
    def reconcile_all(self, sim_vehicles: List[SimulationVehicle], veh_plans: Dict[int, VehiclePlan], 
                      solver_results: Dict[int, VehiclePlan], current_sim_time: int, 
                      routing_engine: NetworkBase, rq_dict: Dict[Any, PlanRequest], 
                      active_rids: Set[Any], backup_solutions: Optional[List[Dict[int, VehiclePlan]]] = None) -> Dict[int, VehiclePlan]:
        """Reconcile solver results for all vehicles.
        
        :param sim_vehicles: list of live SimulationVehicle objects
        :param veh_plans: dict vid -> current VehiclePlan (fallback)
        :param solver_results: dict vid -> VehiclePlan from solver
        :param current_sim_time: current simulation time
        :param routing_engine: routing engine for travel time queries
        :param rq_dict: dict rid -> PlanRequest of currently active requests
        :param active_rids: set of rids that are still valid (not timed out/cancelled)
        :param backup_solutions: optional list of alternative solution dicts (from solution pool)
        :return: dict vid -> reconciled VehiclePlan (only vehicles with changed plans)
        """
        reconciled_plans = {}
        veh_dict = {veh.vid: veh for veh in sim_vehicles}
        
        for vid, solver_plan in solver_results.items():
            veh_obj = veh_dict.get(vid)
            if not veh_obj:
                LOG.warning(f"Vehicle {vid} in solver results not found in sim_vehicles.")
                continue
                
            current_plan = veh_plans.get(vid)
            reconciled = self._reconcile_vehicle(
                veh_obj, solver_plan, current_sim_time, routing_engine, rq_dict, active_rids, current_plan=current_plan
            )
            
            if reconciled is not None:
                reconciled_plans[vid] = reconciled
            elif backup_solutions:
                # Try backup solutions
                success = False
                for backup in backup_solutions:
                    if vid in backup:
                        backup_plan = backup[vid]
                        reconciled_backup = self._reconcile_vehicle(
                            veh_obj, backup_plan, current_sim_time, routing_engine, rq_dict, active_rids, current_plan=current_plan
                        )
                        if reconciled_backup is not None:
                            reconciled_plans[vid] = reconciled_backup
                            success = True
                            break
                if not success:
                    LOG.info(f"All backups failed for vehicle {vid}. Keeping current plan.")
            else:
                LOG.info(f"Reconciliation failed for vehicle {vid} and no backups available. Keeping current plan.")
                
        return reconciled_plans
    
    def _reconcile_vehicle(self, veh_obj: SimulationVehicle, solver_plan: Optional[VehiclePlan], 
                           current_sim_time: int, routing_engine: NetworkBase, 
                           rq_dict: Dict[Any, PlanRequest], active_rids: Set[Any],
                           current_plan: Optional[VehiclePlan] = None) -> Optional[VehiclePlan]:
        """Reconcile a single vehicle's plan.
        
        :return: reconciled VehiclePlan if successful, None if plan should be kept as-is
        """
        if solver_plan is None:
            if veh_obj.pax:
                LOG.warning(f"Solver returned None plan for vehicle {veh_obj.vid} with onboard pax. Keeping current plan.")
                return None
            return VehiclePlan(veh_obj, current_sim_time, routing_engine, [])

        onboard_rids = {rq.get_rid_struct() for rq in veh_obj.pax}
        
        filtered_stops = self._filter_plan_stops(solver_plan.list_plan_stops, active_rids, onboard_rids, rq_dict=rq_dict)
        
        # If vehicle is actively boarding/locked, ensure active stop is retained/merged at the head of the plan
        if veh_obj.assigned_route:
            ca = veh_obj.assigned_route[0]
            if (ca.status in G_LOCK_DURATION_STATUS or ca.locked) and current_plan and current_plan.list_plan_stops:
                filtered_active_stops = self._filter_plan_stops([current_plan.list_plan_stops[0]], active_rids, onboard_rids, rq_dict=rq_dict)
                if filtered_active_stops:
                    active_stop = filtered_active_stops[0]
                    if not filtered_stops or filtered_stops[0].get_pos() != ca.destination_pos:
                        filtered_stops = [active_stop] + filtered_stops
                    else:
                        # Merge active_stop into solver's head stop at the same position
                        head_stop = filtered_stops[0]
                        merged_boarding = list(dict.fromkeys(active_stop.get_list_boarding_rids() + head_stop.get_list_boarding_rids()))
                        merged_alighting = list(dict.fromkeys(active_stop.get_list_alighting_rids() + head_stop.get_list_alighting_rids()))
                        new_b_dict = {}
                        if merged_boarding:
                            new_b_dict[1] = merged_boarding
                        if merged_alighting:
                            new_b_dict[-1] = merged_alighting
                        head_stop.boarding_dict = new_b_dict
                        if rq_dict is not None:
                            head_stop.change_nr_pax = sum(rq_dict[r].nr_pax for r in merged_boarding if r in rq_dict) - \
                                                      sum(rq_dict[r].nr_pax for r in merged_alighting if r in rq_dict)
                        head_stop.earliest_pickup_time_dict.update(active_stop.earliest_pickup_time_dict)
                        head_stop.latest_pickup_time_dict.update(active_stop.latest_pickup_time_dict)
                        head_stop.latest_arrival_time_dict.update(active_stop.latest_arrival_time_dict)
                        head_stop.max_trip_time_dict.update(active_stop.max_trip_time_dict)
                        head_stop.locked = active_stop.locked or head_stop.locked
        
        filtered_stops, missing_rids = self._ensure_onboard_dropoffs(filtered_stops, onboard_rids, active_rids=active_rids)
        
        if missing_rids:
            LOG.error(f"Fatal inconsistency: Vehicle {veh_obj.vid} has onboard passengers {missing_rids} missing dropoffs in solver plan.")
            return None
            
        # Ensure all onboard passengers have a valid numerical pu_time
        for rq in veh_obj.pax:
            if getattr(rq, 'pu_time', None) is None:
                rq.pu_time = current_sim_time

        new_plan = VehiclePlan(veh_obj, current_sim_time, routing_engine, filtered_stops, copy=False)
        
        if new_plan.feasible:
            return new_plan
        else:
            LOG.info(f"Reconciled plan for vehicle {veh_obj.vid} is infeasible.")
            return None
    
    def _filter_plan_stops(self, plan_stops: List[PlanStop], active_rids: Set[Any], onboard_rids: Set[Any],
                           rq_dict: Dict[Any, PlanRequest] = None) -> List[PlanStop]:
        """Filter PlanStops to remove references to invalid requests.
        
        For each PlanStop:
        - Remove boarding rids that are no longer active
        - Remove alighting rids that are no longer active AND not onboard
        - Keep alighting rids for onboard passengers even if cancelled
          (they need to be dropped off somewhere)
        - Recalculate change_nr_pax based on filtered rids
        - If a PlanStop becomes empty (no boarding, no alighting, no charging,
          not locked), remove it
        
        :return: filtered list of PlanStop copies
        """
        filtered = []
        valid_boarding = set()
        
        for stop in plan_stops:
            new_stop = stop.copy()
            
            # Boarding processing
            board_list = new_stop.get_list_boarding_rids()
            new_board_list = [rid for rid in board_list if rid in active_rids]
            
            for rid in new_board_list:
                valid_boarding.add(rid)
                
            # Alighting processing
            alight_list = new_stop.get_list_alighting_rids()
            new_alight_list = [rid for rid in alight_list if (rid in onboard_rids) or (rid in active_rids and rid in valid_boarding)]
            
            # Update boarding dict
            new_boarding_dict = {}
            if new_board_list:
                new_boarding_dict[1] = new_board_list
            if new_alight_list:
                new_boarding_dict[-1] = new_alight_list
            new_stop.boarding_dict = new_boarding_dict
            
            # Recalculate change_nr_pax from filtered rids
            if rq_dict is not None and (len(board_list) != len(new_board_list) or len(alight_list) != len(new_alight_list)):
                nr_boarding = sum(rq_dict[rid].nr_pax for rid in new_board_list if rid in rq_dict)
                nr_alighting = sum(rq_dict[rid].nr_pax for rid in new_alight_list if rid in rq_dict)
                new_stop.change_nr_pax = nr_boarding - nr_alighting
            
            # Update meta-information based on kept rids
            kept_rids = set(new_board_list + new_alight_list)
            
            for d in [new_stop.max_trip_time_dict, new_stop.latest_arrival_time_dict,
                      new_stop.earliest_pickup_time_dict, new_stop.latest_pickup_time_dict]:
                if d:
                    keys_to_remove = [k for k in d.keys() if k not in kept_rids]
                    for k in keys_to_remove:
                        del d[k]
            
            # Keep stop if it has meaningful activity
            has_pax_activity = bool(new_board_list) or bool(new_alight_list)
            has_infra_activity = new_stop.get_charging_power() > 0
            is_protected = new_stop.is_locked() or new_stop.is_locked_end()
            # Non-boarding states (repo, inactive, reservation) should be kept
            is_non_boarding_task = new_stop.get_state() not in (G_PLANSTOP_STATES.BOARDING, G_PLANSTOP_STATES.MIXED)
            
            if has_pax_activity or has_infra_activity or is_protected or is_non_boarding_task:
                filtered.append(new_stop)
                
        return filtered
    
    def _ensure_onboard_dropoffs(self, filtered_stops: List[PlanStop], onboard_rids: Set[Any], 
                                 active_rids: Set[Any] = None) -> Tuple[List[PlanStop], Set[Any]]:
        """Verify all onboard passengers have dropoff stops.
        
        :return: (filtered_stops, missing_rids) where missing_rids is a set
                 of rids onboard but with no dropoff in the plan
        """
        planned_dropoffs = set()
        for stop in filtered_stops:
            planned_dropoffs.update(stop.get_list_alighting_rids())
            
        valid_onboard = set(onboard_rids)
        missing = valid_onboard - planned_dropoffs
        return filtered_stops, missing
