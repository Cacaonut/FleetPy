import unittest
import sys
import os
from typing import List, Dict, Any, Set, Optional

# Ensure FleetPy root is in sys.path
fleetpy_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if fleetpy_root not in sys.path:
    sys.path.insert(0, fleetpy_root)

from src.fleetctrl.AsyncRidePoolingBatchOptimizationFleetControl import QueueingRPBOProxy
from src.fleetctrl.reconciliation.PlanReconciler import PlanReconciler
from src.fleetctrl.planning.VehiclePlan import PlanStop, VehiclePlan
from src.misc.globals import VRL_STATES, G_PLANSTOP_STATES


class MockRPBO:
    """Mock real RPBO module to verify event interception, draining, replay, and pass-through."""
    def __init__(self):
        self.calls: List[tuple] = []
        self.added_requests: List[tuple] = []
        self.deleted_requests: List[Any] = []
        self.assigned_requests: List[Any] = []
        self.boarding_events: List[tuple] = []
        self.alighting_events: List[tuple] = []
        self.assignments: Dict[int, tuple] = {}
        self.locked_requests: Dict[Any, int] = {}
        self.del_requests: List[Any] = []
        self.time_constraint_changes: List[tuple] = []
        self.deleted_vehicle_db_entries: List[int] = []
        self.custom_attribute = "initial_value"

    def add_new_request(self, rid, prq, **kwargs):
        self.calls.append(('add_new_request', (rid, prq), kwargs))
        self.added_requests.append((rid, prq, kwargs))

    def delete_request(self, rid):
        self.calls.append(('delete_request', (rid,), {}))
        self.deleted_requests.append(rid)

    def set_request_assigned(self, rid):
        self.calls.append(('set_request_assigned', (rid,), {}))
        self.assigned_requests.append(rid)

    def set_database_in_case_of_boarding(self, rid, vid):
        self.calls.append(('set_database_in_case_of_boarding', (rid, vid), {}))
        self.boarding_events.append((rid, vid))

    def set_database_in_case_of_alighting(self, rid, vid):
        self.calls.append(('set_database_in_case_of_alighting', (rid, vid), {}))
        self.alighting_events.append((rid, vid))

    def set_assignment(self, vid, plan, **kwargs):
        self.calls.append(('set_assignment', (vid, plan), kwargs))
        self.assignments[vid] = (plan, kwargs)

    def lock_request_to_vehicle(self, rid, vid):
        self.calls.append(('lock_request_to_vehicle', (rid, vid), {}))
        self.locked_requests[rid] = vid

    def delRequest(self, rid):
        self.calls.append(('delRequest', (rid,), {}))
        self.del_requests.append(rid)

    def register_change_in_time_constraints(self, rid, *args, **kwargs):
        self.calls.append(('register_change_in_time_constraints', (rid, *args), kwargs))
        self.time_constraint_changes.append((rid, args, kwargs))

    def delete_vehicle_database_entries(self, vid):
        self.calls.append(('delete_vehicle_database_entries', (vid,), {}))
        self.deleted_vehicle_db_entries.append(vid)

    def query_solver_status(self, status_code: str) -> str:
        return f"solver_ok_{status_code}"

    def compute_objective_value(self, x: float, y: float) -> float:
        return x * 2.0 + y


class TestQueueingRPBOProxy(unittest.TestCase):
    """Unit tests for QueueingRPBOProxy covering event interception, queue draining, event replay, and pass-through."""

    def setUp(self):
        self.mock_real = MockRPBO()
        self.proxy = QueueingRPBOProxy(self.mock_real)

    def test_mutating_methods_interception(self):
        """Test that all mutating methods are intercepted and queued without immediate execution on real module."""
        # 1. add_new_request
        self.proxy.add_new_request(101, "prq_101", consider_for_global_optimisation=True)
        # 2. delete_request
        self.proxy.delete_request(202)
        # 3. set_request_assigned
        self.proxy.set_request_assigned(101)
        # 4. set_database_in_case_of_boarding
        self.proxy.set_database_in_case_of_boarding(101, 0)
        # 5. set_database_in_case_of_alighting
        self.proxy.set_database_in_case_of_alighting(101, 0)
        # 6. set_assignment
        mock_plan = "mock_vehicle_plan_obj"
        self.proxy.set_assignment(0, mock_plan, is_batch=True)
        # 7. lock_request_to_vehicle
        self.proxy.lock_request_to_vehicle(101, 0)

        # Additional queued methods
        self.proxy.delRequest(303)
        self.proxy.register_change_in_time_constraints(101, 120.0, mode="extend")
        self.proxy.delete_vehicle_database_entries(0)

        # Verify real module has NOT been touched by any mutating calls yet
        self.assertEqual(len(self.mock_real.calls), 0)
        self.assertEqual(len(self.mock_real.added_requests), 0)
        self.assertEqual(len(self.mock_real.deleted_requests), 0)
        self.assertEqual(len(self.mock_real.assigned_requests), 0)
        self.assertEqual(len(self.mock_real.boarding_events), 0)
        self.assertEqual(len(self.mock_real.alighting_events), 0)
        self.assertEqual(len(self.mock_real.assignments), 0)
        self.assertEqual(len(self.mock_real.locked_requests), 0)
        self.assertEqual(len(self.mock_real.del_requests), 0)
        self.assertEqual(len(self.mock_real.time_constraint_changes), 0)
        self.assertEqual(len(self.mock_real.deleted_vehicle_db_entries), 0)

    def test_drain_events_empties_queue(self):
        """Test that drain_events() retrieves all queued events and empties the internal queue."""
        self.proxy.add_new_request(101, "prq_101")
        self.proxy.set_request_assigned(101)
        self.proxy.delete_request(202)

        # First drain should return the 3 queued events in order
        events = self.proxy.drain_events()
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0][0], "add_new_request")
        self.assertEqual(events[0][1], (101, "prq_101"))
        self.assertEqual(events[1][0], "set_request_assigned")
        self.assertEqual(events[1][1], (101,))
        self.assertEqual(events[2][0], "delete_request")
        self.assertEqual(events[2][1], (202,))

        # Subsequent drain should be empty
        subsequent_events = self.proxy.drain_events()
        self.assertEqual(len(subsequent_events), 0)

    def test_replay_events_invokes_methods_on_real_module(self):
        """Test that replay_events(events) correctly invokes the mutating methods on the mock real module."""
        # Queue all 7 specified mutating methods
        self.proxy.add_new_request(101, "prq_101", priority=1)
        self.proxy.delete_request(202)
        self.proxy.set_request_assigned(101)
        self.proxy.set_database_in_case_of_boarding(101, 5)
        self.proxy.set_database_in_case_of_alighting(101, 5)
        self.proxy.set_assignment(5, "plan_5", note="test")
        self.proxy.lock_request_to_vehicle(101, 5)

        events = self.proxy.drain_events()
        self.assertEqual(len(events), 7)

        # Replay events
        self.proxy.replay_events(events)

        # Check that real module received all invocations with proper parameters
        self.assertEqual(len(self.mock_real.calls), 7)
        self.assertEqual(self.mock_real.added_requests, [(101, "prq_101", {"priority": 1})])
        self.assertEqual(self.mock_real.deleted_requests, [202])
        self.assertEqual(self.mock_real.assigned_requests, [101])
        self.assertEqual(self.mock_real.boarding_events, [(101, 5)])
        self.assertEqual(self.mock_real.alighting_events, [(101, 5)])
        self.assertEqual(self.mock_real.assignments[5], ("plan_5", {"note": "test"}))
        self.assertEqual(self.mock_real.locked_requests[101], 5)

    def test_passthrough_non_mutating_attributes_and_methods(self):
        """Test that non-mutating attribute access and method calls are passed through directly."""
        # Direct read access
        self.assertEqual(self.proxy.custom_attribute, "initial_value")
        self.assertEqual(self.proxy.query_solver_status("active"), "solver_ok_active")
        self.assertEqual(self.proxy.compute_objective_value(3.0, 4.0), 10.0)

        # Verify non-mutating calls are NOT queued
        self.assertEqual(len(self.proxy.drain_events()), 0)

        # Direct write access passes through to real module
        self.proxy.custom_attribute = "modified_value"
        self.assertEqual(self.mock_real.custom_attribute, "modified_value")
        self.assertEqual(self.proxy.custom_attribute, "modified_value")

        # Property access to real_module
        self.assertIs(self.proxy.real_module, self.mock_real)


class MockPax:
    """Mock passenger object for PlanReconciler tests."""
    def __init__(self, rid: Any, nr_pax: int = 1, pu_time: float = 0.0):
        self.rid = rid
        self._nr_pax = nr_pax
        self.pu_time = pu_time

    def get_rid_struct(self) -> Any:
        return self.rid

    @property
    def nr_pax(self) -> int:
        return self._nr_pax

    @property
    def is_parcel(self) -> bool:
        return False


class MockRoutingEngine:
    """Mock routing engine returning 3-tuple (costs, travel_time, travel_distance)."""
    def return_travel_costs_1to1(self, pos1: tuple, pos2: tuple, customized_section_cost_function=None):
        return (60.0, 60.0, 1000.0)


class MockVehicle:
    """Mock SimulationVehicle for PlanReconciler tests."""
    def __init__(self, vid: int = 0, pos: tuple = (0, 0, 0), pax: Optional[List[MockPax]] = None,
                 soc: float = 1.0, max_pax: int = 4, max_parcels: int = 0):
        self.vid = vid
        self.op_id = 0
        self.pos = pos
        self.pax = list(pax) if pax else []
        self.status = VRL_STATES.IDLE
        self.assigned_route = []
        self.soc = soc
        self.max_pax = max_pax
        self.max_parcels = max_parcels
        self.battery_size = 50.0
        self.soc_per_m = 0.0001
        self.cl_start_time = 0

    def get_nr_pax_without_currently_boarding(self) -> int:
        return sum(rq.nr_pax for rq in self.pax)

    def get_nr_parcels_without_currently_boarding(self) -> int:
        return sum(getattr(rq, 'nr_parcels', 0) for rq in self.pax)

    def compute_soc_consumption(self, tdist: float) -> float:
        return (tdist / 1000.0) * self.soc_per_m

    def compute_soc_charging(self, power: float, duration: float) -> float:
        return (power * duration) / (self.battery_size * 3600.0)


class TestPlanReconciler(unittest.TestCase):
    """Unit tests for PlanReconciler covering empty plans, onboard continuity, cancelled requests, and fatal inconsistencies."""

    def setUp(self):
        self.reconciler = PlanReconciler()
        self.routing_engine = MockRoutingEngine()

    def test_empty_solver_plan_for_idle_vehicle_without_pax(self):
        """Test that an empty solver plan (None or empty VehiclePlan) for an idle vehicle without passengers returns an empty VehiclePlan."""
        idle_veh = MockVehicle(vid=0, pos=(0, 0, 0), pax=[])

        # 1. solver_plan is None
        reconciled_none = self.reconciler._reconcile_vehicle(
            veh_obj=idle_veh,
            solver_plan=None,
            current_sim_time=100,
            routing_engine=self.routing_engine,
            rq_dict={},
            active_rids=set()
        )
        self.assertIsNotNone(reconciled_none)
        self.assertIsInstance(reconciled_none, VehiclePlan)
        self.assertEqual(len(reconciled_none.list_plan_stops), 0)
        self.assertTrue(reconciled_none.is_feasible())

        # 2. solver_plan is an empty VehiclePlan instance
        empty_plan = VehiclePlan(idle_veh, 100, self.routing_engine, [])
        reconciled_empty = self.reconciler._reconcile_vehicle(
            veh_obj=idle_veh,
            solver_plan=empty_plan,
            current_sim_time=100,
            routing_engine=self.routing_engine,
            rq_dict={},
            active_rids=set()
        )
        self.assertIsNotNone(reconciled_empty)
        self.assertEqual(len(reconciled_empty.list_plan_stops), 0)
        self.assertTrue(reconciled_empty.is_feasible())

        # 3. reconcile_all batch call
        all_reconciled = self.reconciler.reconcile_all(
            sim_vehicles=[idle_veh],
            veh_plans={0: None},
            solver_results={0: None},
            current_sim_time=100,
            routing_engine=self.routing_engine,
            rq_dict={},
            active_rids=set()
        )
        self.assertIn(0, all_reconciled)
        self.assertEqual(len(all_reconciled[0].list_plan_stops), 0)

    def test_onboard_passenger_continuity(self):
        """Test onboard passenger continuity: when vehicle has passenger 101 onboard, its dropoff is retained and pickup is stripped."""
        # Vehicle has passenger 101 already onboard
        veh = MockVehicle(vid=0, pos=(0, 0, 0), pax=[MockPax(rid=101, pu_time=50.0)])

        # Solver plan computed before pickup had both pickup and dropoff PlanStops for 101
        stop_pickup = PlanStop(position=(1, 0, 0), boarding_dict={1: [101]}, change_nr_pax=1)
        stop_dropoff = PlanStop(position=(2, 0, 0), boarding_dict={-1: [101]}, change_nr_pax=-1)
        solver_plan = VehiclePlan(MockVehicle(vid=0, pos=(0, 0, 0), pax=[]), 100, self.routing_engine, [stop_pickup, stop_dropoff])

        active_rids = {101}
        onboard_rids = {101}

        # Verify _filter_plan_stops directly
        filtered = self.reconciler._filter_plan_stops([stop_pickup, stop_dropoff], active_rids, onboard_rids)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].get_list_boarding_rids(), [])
        self.assertEqual(filtered[0].get_list_alighting_rids(), [101])

        # Verify full _reconcile_vehicle
        reconciled = self.reconciler._reconcile_vehicle(
            veh_obj=veh,
            solver_plan=solver_plan,
            current_sim_time=100,
            routing_engine=self.routing_engine,
            rq_dict={},
            active_rids=active_rids
        )
        self.assertIsNotNone(reconciled)
        self.assertEqual(len(reconciled.list_plan_stops), 1)
        self.assertEqual(reconciled.list_plan_stops[0].get_list_alighting_rids(), [101])
        self.assertEqual(reconciled.list_plan_stops[0].get_list_boarding_rids(), [])
        self.assertTrue(reconciled.is_feasible())

    def test_cancelled_request_filtering(self):
        """Test cancelled request filtering: when request 202 is cancelled (not in active_rids), its pickup and dropoff PlanStops are removed."""
        veh = MockVehicle(vid=0, pos=(0, 0, 0), pax=[])

        # Plan contains stops for active request 101 and cancelled request 202
        stop1_pickup_both = PlanStop(position=(1, 0, 0), boarding_dict={1: [101, 202]}, change_nr_pax=2)
        stop2_dropoff_202 = PlanStop(position=(2, 0, 0), boarding_dict={-1: [202]}, change_nr_pax=-1)
        stop3_dropoff_101 = PlanStop(position=(3, 0, 0), boarding_dict={-1: [101]}, change_nr_pax=-1)

        solver_plan = VehiclePlan(veh, 100, self.routing_engine, [stop1_pickup_both, stop2_dropoff_202, stop3_dropoff_101])

        # Only 101 is active, 202 was cancelled
        active_rids = {101}
        onboard_rids = set()

        # 1. Test _filter_plan_stops directly
        filtered = self.reconciler._filter_plan_stops(solver_plan.list_plan_stops, active_rids, onboard_rids)
        # stop2 should be completely removed, stop1 should only contain 101, stop3 should only contain 101
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0].get_list_boarding_rids(), [101])
        self.assertEqual(filtered[0].get_change_nr_pax(), 1)
        self.assertEqual(filtered[1].get_list_alighting_rids(), [101])
        self.assertEqual(filtered[1].get_change_nr_pax(), -1)

        # 2. Test full reconciliation
        reconciled = self.reconciler._reconcile_vehicle(
            veh_obj=veh,
            solver_plan=solver_plan,
            current_sim_time=100,
            routing_engine=self.routing_engine,
            rq_dict={},
            active_rids=active_rids
        )
        self.assertIsNotNone(reconciled)
        self.assertEqual(len(reconciled.list_plan_stops), 2)
        self.assertEqual(reconciled.list_plan_stops[0].get_list_boarding_rids(), [101])
        self.assertEqual(reconciled.list_plan_stops[1].get_list_alighting_rids(), [101])
        self.assertTrue(reconciled.is_feasible())

        # 3. Test when the entire plan only had cancelled request 202
        solo_pu_202 = PlanStop(position=(1, 0, 0), boarding_dict={1: [202]}, change_nr_pax=1)
        solo_do_202 = PlanStop(position=(2, 0, 0), boarding_dict={-1: [202]}, change_nr_pax=-1)
        solo_plan_202 = VehiclePlan(veh, 100, self.routing_engine, [solo_pu_202, solo_do_202])

        reconciled_solo = self.reconciler._reconcile_vehicle(
            veh_obj=veh,
            solver_plan=solo_plan_202,
            current_sim_time=100,
            routing_engine=self.routing_engine,
            rq_dict={},
            active_rids=set()  # 202 is cancelled
        )
        self.assertIsNotNone(reconciled_solo)
        self.assertEqual(len(reconciled_solo.list_plan_stops), 0)
        self.assertTrue(reconciled_solo.is_feasible())

    def test_fatal_inconsistency_detection(self):
        """Test fatal inconsistency detection: when vehicle has passenger 303 onboard but solver plan does NOT have dropoff for 303 -> reconciler rejects plan (returns None)."""
        # Vehicle has passenger 303 onboard
        veh = MockVehicle(vid=0, pos=(0, 0, 0), pax=[MockPax(rid=303, pu_time=40.0)])

        # Case A: solver_plan is None with passenger 303 onboard
        reconciled_none = self.reconciler._reconcile_vehicle(
            veh_obj=veh,
            solver_plan=None,
            current_sim_time=100,
            routing_engine=self.routing_engine,
            rq_dict={},
            active_rids=set()
        )
        self.assertIsNone(reconciled_none, "Reconciler must reject None plan when passenger 303 is onboard.")

        # Case B: solver_plan has stops for another request 404, but lacks dropoff for 303
        stop_pu_404 = PlanStop(position=(1, 0, 0), boarding_dict={1: [404]}, change_nr_pax=1)
        stop_do_404 = PlanStop(position=(2, 0, 0), boarding_dict={-1: [404]}, change_nr_pax=-1)
        plan_without_303 = VehiclePlan(MockVehicle(vid=0, pos=(0, 0, 0), pax=[]), 100, self.routing_engine, [stop_pu_404, stop_do_404])

        reconciled_missing_dropoff = self.reconciler._reconcile_vehicle(
            veh_obj=veh,
            solver_plan=plan_without_303,
            current_sim_time=100,
            routing_engine=self.routing_engine,
            rq_dict={},
            active_rids={404}
        )
        self.assertIsNone(reconciled_missing_dropoff, "Reconciler must reject plan that omits onboard passenger 303 dropoff.")

        # Case C: _ensure_onboard_dropoffs directly reports missing rids
        _, missing = self.reconciler._ensure_onboard_dropoffs([stop_pu_404, stop_do_404], onboard_rids={303})
        self.assertEqual(missing, {303})

        # Case D: reconcile_all does not include vehicle in result when rejected
        reconciled_all = self.reconciler.reconcile_all(
            sim_vehicles=[veh],
            veh_plans={0: None},
            solver_results={0: plan_without_303},
            current_sim_time=100,
            routing_engine=self.routing_engine,
            rq_dict={},
            active_rids={404}
        )
        self.assertNotIn(0, reconciled_all, "Inconsistent plan should be excluded from reconcile_all output.")


if __name__ == '__main__':
    unittest.main()
