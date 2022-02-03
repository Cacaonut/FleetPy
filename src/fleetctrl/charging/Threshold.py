import logging

from src.fleetctrl.charging.ChargingBase import ChargingBase
from src.simulation.StationaryProcess import ChargingProcess
from src.fleetctrl.planning.VehiclePlan import PlanStop
from src.misc.globals import *

LOG = logging.getLogger(__name__)


class ChargingThresholdPublicInfrastructure(ChargingBase):
    """ this strategy looks through all fleet vehicles an triggers charging tasks in case the soc within a planned route
    of a vehicle drops below a threshold (G_OP_APS_SOC)
    the closest charging station possible from this position is considered for charging
    in case multiple charging operators are present, the offer closest to postion is selected (also with depots)"""
    def __init__(self, fleetctrl, operator_attributes):
        super().__init__(fleetctrl, operator_attributes)
        self.soc_threshold = operator_attributes.get(G_OP_APS_SOC, 0.1)

    def time_triggered_charging_processes(self, sim_time):
        for veh_obj in self.fleetctrl.sim_vehicles:
            # do not consider inactive vehicles
            if veh_obj.status in {VRL_STATES.OUT_OF_SERVICE, VRL_STATES.BLOCKED_INIT}:
                continue
            current_plan = self.fleetctrl.veh_plans[veh_obj.vid]
            is_charging_required = False
            last_time = sim_time
            last_pos = veh_obj.pos
            last_soc = veh_obj.soc
            if current_plan.list_plan_stops:
                last_pstop = current_plan.list_plan_stops[-1]
                pstop_task = last_pstop.stationary_task
                if pstop_task is not None and not isinstance(pstop_task, ChargingProcess):
                    last_soc, _ = current_plan.list_plan_stops[-1].get_planned_arrival_and_departure_soc()
                    if last_soc < self.soc_threshold and not last_pstop.is_inactive():
                        _, last_time = last_pstop.get_planned_arrival_and_departure_time()
                        last_pos = last_pstop.get_pos()
                        is_charging_required = True
            elif veh_obj.soc < self.soc_threshold:
                is_charging_required = True

            if is_charging_required is True:
                best_charging_poss = None
                best_ch_op = None
                for ch_op in self.all_charging_infra:
                    charging_possibilities = ch_op.get_charging_slots(sim_time, veh_obj, last_time, last_pos, last_soc, 1.0, 1, 1)
                    if len(charging_possibilities) > 0:
                        ch_op_best = min(charging_possibilities, key=lambda x:x[5])
                        if best_charging_poss is None or ch_op_best[5] < best_charging_poss[5]:
                            best_charging_poss = ch_op_best
                            best_ch_op = ch_op
                if best_charging_poss is not None:
                    (station_id, socket_id, possible_start_time, possible_end_time, desired_veh_soc, tt, dis) = best_charging_poss
                    booking = best_ch_op.book_station(sim_time, veh_obj, station_id, socket_id, possible_start_time, possible_end_time)
                    station = best_ch_op.station_by_id[station_id]
                    ps = PlanStop(station.pos, {}, {}, {}, {}, {}, locked=True, stationary_task=booking,
                                status=VRL_STATES.CHARGING)
                    current_plan.add_plan_stop(ps, veh_obj, sim_time, self.routing_engine)
                    self.fleetctrl.lock_current_vehicle_plan(veh_obj.vid)
                    self.fleetctrl.assign_vehicle_plan(veh_obj, current_plan, sim_time)
