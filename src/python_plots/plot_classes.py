import os
from multiprocessing import Process
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import typing as tp
from pathlib import Path
import contextily as ctx
from pyproj import Transformer
from datetime import datetime, timedelta
import geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import substring
from src.misc.globals import *
import pandas as pd
import warnings

FIG_SIZE = (15,10)
# Number of historical points to be displayed on the x-axis
PLOT_LENGTH = 200
# Delay between frames in milliseconds
REALTIME_UPDATE_INTERVAL = 200
VEHICLE_POINT_SIZE = 12
CTX_PROVIDER = ctx.providers.CartoDB.Positron# ctx.providers.Stamen.TonerLite
BOARDER_SIZE = 1000

import matplotlib.colors
tab20 = plt.cm.get_cmap('Oranges', 20)
cl = tab20(np.linspace(0.2, 1, 5))
STATUS_COLOR_LIST= ["lightgrey","red","blue","orange","green","dimgrey","purple","beige"]
OCCUPANCY_COLOR_LIST =  list(cl) + ['dodgerblue'] + ['dimgrey']

# SMALL_SIZE = 8
# MEDIUM_SIZE = 18
# BIGGER_SIZE = 20

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


class PyPlot(Process):

    def __init__(self, nw_dir, shared_dict: dict, plot_folder: str = None, plot_extent=None, output_dir=None):
        """ Class for plotting real time information

        :param nw_dir:      network directory, where the background map is/will be saved
        :param shared_dict: a dictionary object for sharing real time information
        :param plot_folder: full path of the folder for saving the animation. If provided, each frame is saved
                            in the location.
        :param plot_extent: Tuple of (lon_1, lon_2, lat_1, lat_2) marking the left bottom and top right boundary of
                            the map
        """

        super().__init__()
        self.bg_map_path = os.path.join(nw_dir, "downloaded_map.tif")
        self.shared_dict: dict = shared_dict
        self.plot_folder: Path = Path(plot_folder) if plot_folder else None

        self.fig, self.grid_spec, self.axes = None, None, None
        self.plot_axes = []
        self.map_ax = None
        self.plot_extent = plot_extent
        x, y = self.convert_lat_lon(plot_extent[2:4], plot_extent[0:2])
        x[0], y[0] = x[0] - BOARDER_SIZE, y[0] - BOARDER_SIZE
        x[1], y[1] = x[1] + BOARDER_SIZE, y[1] + BOARDER_SIZE
        self.plot_extent_3857 = x + y
        # Download the map image using the extent
        _, bbox = ctx.bounds2raster(x[0], y[0], x[1], y[1], path=self.bg_map_path, source=CTX_PROVIDER)
        #self.plot_extent_3857 = bbox
        
        self._times = []
        self._pax_counts = {}
        self._state_counts = {}
        self._avg_occ = []
        self._avg_wait_time = []
        self._avg_ride_time = []
        self._avg_detour_time = []
        self._accepted_requests = []
        self._rejected_requests = []
        self._timed_out_requests = []
        self._queue_lengths = []
        self._response_times = []
        self._tick_durations = []
        self._step_budgets = []
        self._reopt_budgets = []
        self._lag_flags = []
        
        self._key_to_plot_func = {
            'status_count': self._create_status_count_plot,
            'occupancy_count': self._create_occ_count_plot,
            'occupancy_average': self._create_avg_occ_plot,
            'occupancy_stack_chart': self._create_occ_stack_plot,
            'waiting_time_average': self._create_avg_wait_time_plot,
            'ride_time_average': self._create_avg_ride_time_plot,
            'detour_time_average': self._create_avg_detour_time_plot,
            'service_rate': self._create_service_rate_stack_plot,
            'realtime_architecture': self._create_realtime_architecture_plot,
            'queue_length': self._create_queue_length_plot,
            'realtime_lag': self._create_realtime_lag_plot,
            'queue_list': self._create_queue_list_plot
        }

        self.line_alignment = None
        # load PT Line alignment parameters
        if output_dir is not None:
            scenario_parameters, list_operator_attributes, _ = load_scenario_inputs(output_dir)
            dir_names = get_directory_dict(scenario_parameters, list_operator_attributes)

            if scenario_parameters.get(G_PT_SCHEDULE_F) and G_DIR_PT in dir_names:
                schedules = pd.read_csv(os.path.join(dir_names[G_DIR_PT], scenario_parameters[G_PT_SCHEDULE_F]))
                # TODO: to generalize for more than 1 line
                for key, _ in schedules.groupby(["LINE", "line_vehicle_id", "vehicle_type"]):
                    self.line_id = key[0]
                    break
                alignment_file = scenario_parameters.get(G_PT_ALIGNMENT_F, 0)
                self.line_alignment: LineString = gpd.read_file(
                    os.path.join(dir_names[G_DIR_PT], alignment_file.format(line_id=self.line_id)))['geometry'].iloc[0]

                self.pt_fixed_length_km: float = scenario_parameters.get(G_PT_FIXED_LENGTH, None)  # km
                self.crs_km = "EPSG:32632"

                line_alignment_km: LineString = self.convert_lat_lon_line(self.line_alignment, from_epsg='epsg:4326',
                                                                          to_epsg=self.crs_km, always_xy=True)
                split_point = self.pt_fixed_length_km * 1000 / line_alignment_km.length
                first_part_km = substring(self.line_alignment, 0, split_point, normalized=True)
                second_part_km = substring(self.line_alignment, split_point, 1, normalized=True)

                lon, lat = first_part_km.xy
                self.first_part_x, self.first_part_y = self.convert_lat_lon(lat, lon, from_epsg='epsg:4326')
                lon, lat = second_part_km.xy
                self.second_part_x, self.second_part_y = self.convert_lat_lon(lat, lon, from_epsg='epsg:4326')

    def convert_lat_lon(self, lats: list, lons: list, to_epsg: str = "epsg:3857", from_epsg: str = 'epsg:4326'):
        proj_transformer = Transformer.from_proj(from_epsg, to_epsg)
        x, y = proj_transformer.transform(lats, lons)
        return list(x), list(y)

    def convert_lat_lon_line(self, line: LineString, to_epsg: str = "epsg:3857", from_epsg: str = 'epsg:4326',
                             always_xy:bool = False):
        lats, lons = line.xy
        proj_transformer = Transformer.from_proj(from_epsg, to_epsg, always_xy=always_xy)
        x, y = proj_transformer.transform(lats, lons)
        return LineString(zip(x, y))

    def _get_ax(self, target):
        if isinstance(target, int):
            return self.plot_axes[target] if target < len(self.plot_axes) else self.axes[target]
        return target

    def generate_plot_axes(self):
        has_second_column = any(self.shared_dict.get(f"plot_{i}") is not None for i in [4, 5, 6])
        if has_second_column:
            fig = plt.figure(1, figsize=(18, 9.5), tight_layout=True)
            gs = gridspec.GridSpec(3, 3, figure=fig, width_ratios=[2.0, 1.0, 1.0])
            gs.update(wspace=0.18, hspace=0.45)
            self.plot_axes = [
                plt.subplot(gs[0, 1]), plt.subplot(gs[1, 1]), plt.subplot(gs[2, 1]),
                plt.subplot(gs[0, 2]), plt.subplot(gs[1, 2]), plt.subplot(gs[2, 2])
            ]
            self.map_ax = plt.subplot(gs[:, 0])
        else:
            fig = plt.figure(1, figsize=FIG_SIZE, tight_layout=True)
            gs = gridspec.GridSpec(3, 3, figure=fig)
            gs.update(wspace=0.025, hspace=0.5)
            self.plot_axes = [
                plt.subplot(gs[0, 2]), plt.subplot(gs[1, 2]), plt.subplot(gs[2, 2])
            ]
            self.map_ax = plt.subplot(gs[:, 0:2])
        self.axes = self.plot_axes + [self.map_ax]
        return fig, gs, self.axes

    def draw_plots(self):
        self._times.append(self.shared_dict["sim_time_float"])
        for k, v in self.shared_dict.get("status_counts", {}).items():
            if self._state_counts.get(k) is None:
                self._state_counts[k] = []
            self._state_counts[k].append(v)
        for k, v in self.shared_dict.get("pax_info", {}).items():
            if self._pax_counts.get(k) is None:
                self._pax_counts[k] = []
            self._pax_counts[k].append(v)
        self._avg_occ.append(self.shared_dict.get("avg_pax", 0))
        self._avg_wait_time.append(self.shared_dict.get("avg_wait_time", 0))
        self._avg_ride_time.append(self.shared_dict.get("avg_ride_time", 0))
        self._avg_detour_time.append(self.shared_dict.get("avg_detour_time", 0))
        self._accepted_requests.append(self.shared_dict.get("accepted_users", 0))
        self._rejected_requests.append(self.shared_dict.get("rejected_users", 0))
        self._timed_out_requests.append(self.shared_dict.get("timed_out_users", 0))
        self._queue_lengths.append(self.shared_dict.get("queue_length", 0))
        self._response_times.append(self.shared_dict.get("response_time", 0.0))
        self._tick_durations.append(self.shared_dict.get("tick_duration", 0.0))
        self._step_budgets.append(self.shared_dict.get("step_budget", 1.0))
        self._reopt_budgets.append(self.shared_dict.get("reopt_budget", 1.0))
        self._lag_flags.append(self.shared_dict.get("is_lag", False))

        for idx, ax in enumerate(self.plot_axes):
            plot_key = self.shared_dict.get(f"plot_{idx + 1}")
            if plot_key and plot_key in self._key_to_plot_func:
                self._key_to_plot_func[plot_key](ax)
            
        if self.shared_dict['map_plot'] == "occupancy" and self.shared_dict['parcels']:
            possible_status = ['0 (route)','1','2','3','4','0 (reposition)','idle']
            masks = []
            color_list = OCCUPANCY_COLOR_LIST
            # route
            condition_1 = self.shared_dict["veh_coord_status_df"]["status"] == "route"
            condition_2 = self.shared_dict["veh_coord_status_df"]["parcels"] == 0
            masks.append([a and b for a, b in zip(condition_1, condition_2)])
            
            for i in range(1, 5):
                condition_1 = self.shared_dict["veh_coord_status_df"]["status"] != "idle"
                condition_2 = self.shared_dict["veh_coord_status_df"]["parcels"] == i
                masks.append([a and b for a, b in zip(condition_1, condition_2)])
                
            # repo
            condition_1 = self.shared_dict["veh_coord_status_df"]["status"] == "reposition"
            condition_2 = self.shared_dict["veh_coord_status_df"]["parcels"] == 0
            masks.append([a and b for a, b in zip(condition_1, condition_2)])
            
            masks.append(self.shared_dict["veh_coord_status_df"]["status"] == "idle")
        elif self.shared_dict['map_plot'] == "occupancy" and self.shared_dict['passengers']:
            possible_status = ['0 (route)','1','2','3','4','0 (reposition)','idle']
            masks = []
            color_list = OCCUPANCY_COLOR_LIST
            # route
            condition_1 = self.shared_dict["veh_coord_status_df"]["status"] == "route"
            condition_2 = self.shared_dict["veh_coord_status_df"]["passengers"] == 0
            masks.append([a and b for a, b in zip(condition_1, condition_2)])
            
            for i in range(1, 5):
                condition_1 = self.shared_dict["veh_coord_status_df"]["status"] != "idle"
                condition_2 = self.shared_dict["veh_coord_status_df"]["passengers"] == i
                masks.append([a and b for a, b in zip(condition_1, condition_2)])
                
            # repo
            condition_1 = self.shared_dict["veh_coord_status_df"]["status"] == "reposition"
            condition_2 = self.shared_dict["veh_coord_status_df"]["parcels"] == 0
            masks.append([a and b for a, b in zip(condition_1, condition_2)])
            
            masks.append(self.shared_dict["veh_coord_status_df"]["status"] == "idle")
        elif (self.shared_dict['map_plot'] == "occupancy" 
              and not self.shared_dict['parcels'] 
              and not self.shared_dict['passengers']):
            possible_status = ['0 (route)','1','2','3','4','0 (reposition)','idle']
            masks = []
            color_list = OCCUPANCY_COLOR_LIST
            # route
            condition_1 = self.shared_dict["veh_coord_status_df"]["status"] == "route"
            condition_2 = self.shared_dict["veh_coord_status_df"]["pax"] == 0
            masks.append([a and b for a, b in zip(condition_1, condition_2)])
            
            for i in range(1, 5):
                condition_1 = self.shared_dict["veh_coord_status_df"]["status"] != "idle"
                condition_2 = self.shared_dict["veh_coord_status_df"]["pax"] == i
                masks.append([a and b for a, b in zip(condition_1, condition_2)])
            # repo
            condition_1 = self.shared_dict["veh_coord_status_df"]["status"] == "reposition"
            condition_2 = self.shared_dict["veh_coord_status_df"]["parcels"] == 0
            masks.append([a and b for a, b in zip(condition_1, condition_2)])
                
            masks.append(self.shared_dict["veh_coord_status_df"]["status"] == "idle")
        elif self.shared_dict['map_plot'] == "vehicle_status":
            possible_status = self.shared_dict["possible_status"]
            color_list = STATUS_COLOR_LIST
            masks = []
            for status in possible_status:
                masks.append(self.shared_dict["veh_coord_status_df"]["status"] == status)
        elif self.shared_dict['map_plot'] == "zone":
            possible_status = ['-1','0','1','2','3']
            color_list = STATUS_COLOR_LIST
            vid_zone = [0,0,0,0,1,2,3,4,1,2,3,4]
            masks = []
            for status in range(len(possible_status)):
                masks.append([vid_zone[i] == status for i in range(len(vid_zone))])

        axes = self.axes

        map_ax = self.map_ax

        # Plot PT line alignment
        if self.line_alignment is not None:
            map_ax.plot(self.first_part_x, self.first_part_y, color="black", linewidth=0.5, zorder=1)
            map_ax.plot(self.second_part_x, self.second_part_y, color="black", linewidth=0.5, linestyle="--", zorder=1)

        # Plot the data on the map
        ### Plot the vehicle status statistics
        ###
        map_ax.axis(self.plot_extent_3857)
        map_ax.set_xlim(self.plot_extent_3857[:2])
        map_ax.set_ylim(self.plot_extent_3857[2:])
        passengers = self.shared_dict.get("total_passengers", 0)
        parcels = self.shared_dict.get("total_parcels", 0)
        mode = "passengers" if self.shared_dict.get('passengers') else "parcels"
        if not self.shared_dict.get('parcels') and not self.shared_dict.get('passengers'):
            mode = "pax"
        map_ax.text(0.8, 0.97, f"\n Number of passengers: {passengers}",
                     transform=map_ax.transAxes)
        ctx.add_basemap(map_ax, source=self.bg_map_path)

        vehicle_df = self.shared_dict["veh_coord_status_df"]

        for i, mask in enumerate(masks):
            coords = vehicle_df[mask]["coordinates"].to_list()
            x, y = [], []
            if coords:
                lons, lats = list(zip(*coords))
                x, y = self.convert_lat_lon(lats, lons)
            map_ax.scatter(x, y, s=VEHICLE_POINT_SIZE, label=possible_status[i],color = color_list[i])
        map_ax.legend(loc="lower left")
        map_ax.axis('off')
        rounded_simulation_time = self.shared_dict["simulation_time"] - timedelta(microseconds=self.shared_dict["simulation_time"].microsecond)
        str_simulation_time = "Time: " + rounded_simulation_time.strftime("%H:%M:%S")
        map_ax.set_title(str_simulation_time)

    def save_single_plot(self, datetime_stamp: tp.Union[str, datetime]):
        if self.fig is None:
            self.fig, self.grid_spec, self.axes = self.generate_plot_axes()
        [ax.clear() for ax in self.axes]
        self.draw_plots()
        if self.plot_folder.exists() is False:
            self.plot_folder.mkdir()
        if type(datetime_stamp) == datetime:
            file_name = "plot_{}.png".format(datetime_stamp.strftime("%d-%b-%Y %H-%M-%S"))
        else:
            file_name = "plot_{}.png".format(datetime_stamp)
        plt.savefig(str(self.plot_folder.joinpath(file_name)), bbox_inches = 'tight')

    def __animate(self, i):
        [ax.clear() for ax in self.axes]
        try:
            self.draw_plots()
            if self.grid_spec is not None:
                self.grid_spec.tight_layout(self.fig)
        except KeyError:
            pass

    def run(self):
        def frame():
            i = 0
            while True:
                if self.shared_dict.get("stop", False) is False:
                    i = i + 1
                    yield i
                else:
                    return

        self.fig, self.grid_spec, self.axes = self.generate_plot_axes()
        ani = animation.FuncAnimation(self.fig, self.__animate, interval=REALTIME_UPDATE_INTERVAL)

        # TODO: This is to ignore plot tight_layout warnings
        # warnings.filterwarnings("ignore", category=UserWarning)

        plt.show()
        
    def _create_occ_stack_plot(self, axis_id):
        ax = self._get_ax(axis_id)
        ax.set_title("Occupancy Stack Chart")
        ax.set_ylabel("Number Vehicles")
        list_list_values = [self._pax_counts[k] for k in ['0 (route)','1','2','3','4', '0 (reposition)','idle']]
        ax.stackplot(self._times, *list_list_values,
                     colors=OCCUPANCY_COLOR_LIST,
                     labels = ['0 (route)','1','2','3','4','0 (reposition)','idle' ])
        ax.legend(loc="upper left")
        ax.set_xlabel("Simulation Time [h]")
        ax.tick_params(axis="x", which="both", labelbottom=True)
        ax.tick_params(axis="y", which="both", labelleft=True)
        
        xticks = ax.get_xticks()
        s, e = ax.get_xlim()
        max_xlim = s + 0.9 * (e - s)
        filtered_xticks = [tick for tick in xticks if s <= tick <= max_xlim]
        ax.set_xticks(filtered_xticks)
        
    def _create_status_count_plot(self, axis_id):
        ax = self._get_ax(axis_id)
        ax.set_title("Route Status Counts")
        ax.bar(self.shared_dict["status_counts"].keys(), 
               self.shared_dict["status_counts"].values(),
               color = STATUS_COLOR_LIST)
        ax.set_ylim(0, len(self.shared_dict["veh_coord_status_df"]))
        ax.set_xticks(list(self.shared_dict["status_counts"].keys()))
        xtick_labels = [str(x) if len(str(x)) < 5 else "\n" + str(x) for x in self.shared_dict["status_counts"].keys()]
        ax.set_xticklabels(xtick_labels, rotation=0)
        
    def _create_occ_count_plot(self, axis_id):
        ax = self._get_ax(axis_id)
        ax.set_title("Occupancy Counts")
        ks = ['0 (route)','1','2','3','4', '0 (reposition)','idle']
        ax.bar(ks, 
               [self.shared_dict["pax_info"][k] for k in ks],
               color = OCCUPANCY_COLOR_LIST)
        ax.set_ylim(0, len(self.shared_dict["veh_coord_status_df"]))
        ax.set_xticks(ks)
        xtick_labels = [str(x) if len(str(x)) < 5 else "\n" + str(x) for x in ks]
        ax.set_xticklabels(xtick_labels, rotation=0)
        ax.set_xlabel("Occupancy")
        ax.set_ylabel("Number of Vehicles")
        
    def _create_avg_occ_plot(self, axis_id):
        ax = self._get_ax(axis_id)
        ax.set_title("Average Occupancy")
        ax.plot(self._times, self._avg_occ)
        ax.set_xlabel("Simulation Time [h]")
        ax.set_ylabel("Average Occupancy")
        
    def _create_avg_wait_time_plot(self, axis_id):
        ax = self._get_ax(axis_id)
        ax.set_title("Average Waiting Time")
        ax.set_ylabel("Waiting Time [s]")
        ax.set_xlabel("Simulation Time [h]")
        ax.plot(self._times, self._avg_wait_time)
        
    def _create_avg_ride_time_plot(self, axis_id):
        ax = self._get_ax(axis_id)
        ax.set_title("Average Ride Time")
        ax.set_ylabel("Ride Time [s]")
        ax.set_xlabel("Simulation Time [h]")
        ax.plot(self._times, self._avg_ride_time)
        
    def _create_avg_detour_time_plot(self, axis_id):
        ax = self._get_ax(axis_id)
        ax.set_title("Average Detour Time")
        ax.set_ylabel("Detour Time [s]")
        ax.set_xlabel("Simulation Time [h]")
        ax.plot(self._times, self._avg_detour_time)
        
    def _create_service_rate_stack_plot(self, axis_id):
        ax = self._get_ax(axis_id)
        ax.set_title("Requests States")
        ax.set_ylabel("Number of Requests")
        is_realtime = self.shared_dict.get("is_realtime", False)
        if is_realtime:
            ax.stackplot(self._times, self._accepted_requests, self._rejected_requests, self._timed_out_requests,
                         colors=["green","red","purple"],
                         labels = ["accepted","rejected","timed out"])
        else:
            ax.stackplot(self._times, self._accepted_requests, self._rejected_requests,
                         colors=["green","red"],
                         labels = ["accepted","rejected"])
        ax.legend(loc="upper left")
        ax.set_xlabel("Simulation Time [h]")
        xticks = ax.get_xticks()
        s, e = ax.get_xlim()
        max_xlim = s + 0.9 * (e - s)
        filtered_xticks = [tick for tick in xticks if s <= tick <= max_xlim]
        ax.set_xticks(filtered_xticks)

    def _create_realtime_architecture_plot(self, axis_id):
        ax = self._get_ax(axis_id)
        ax.set_title("Realtime Architecture & Queue", fontsize=12)
        ax.axis("off")
        
        q_len = self.shared_dict.get("queue_length", 0)
        resp_t = self.shared_dict.get("response_time", 0.0)
        tick_d = self.shared_dict.get("tick_duration", 0.0)
        step_b = self.shared_dict.get("step_budget", 1.0)
        is_lag = self.shared_dict.get("is_lag", False)
        
        # Status Badge
        if is_lag:
            status_text = f"TICK LAG ({tick_d:.2f}s > {step_b:.1f}s)"
            status_bg = "#ffcccc"
            status_fg = "#990000"
        else:
            status_text = f"REAL-TIME OK ({tick_d:.2f}s < {step_b:.1f}s)"
            status_bg = "#d4edda"
            status_fg = "#155724"
            
        ax.text(0.5, 0.88, status_text, ha="center", va="center", fontsize=8.5, fontweight="bold",
                color=status_fg, bbox=dict(boxstyle="round,pad=0.25", facecolor=status_bg, edgecolor=status_fg, lw=1.0),
                transform=ax.transAxes)
        
        # Flow Schematic
        ax.text(0.16, 0.55, "Demand\nThread", ha="center", va="center", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#e8f4f8", edgecolor="#2b7bba", lw=1.1),
                transform=ax.transAxes)
                
        ax.annotate("", xy=(0.35, 0.55), xytext=(0.28, 0.55), xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", lw=1.4, color="#444444"))
                    
        q_box_color = "#fff3cd" if q_len == 0 else "#ffeeba"
        q_border = "#856404" if q_len > 0 else "#6c757d"
        ax.text(0.50, 0.55, f"Request\nQueue\n[{q_len} req]", ha="center", va="center", fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", facecolor=q_box_color, edgecolor=q_border, lw=1.3),
                transform=ax.transAxes)
                
        ax.annotate("", xy=(0.72, 0.55), xytext=(0.65, 0.55), xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", lw=1.4, color="#444444"))
                    
        ax.text(0.84, 0.55, "Fleet\nControl", ha="center", va="center", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#e8f4f8", edgecolor="#2b7bba", lw=1.1),
                transform=ax.transAxes)
                
        # Response time label directly beneath the Queue box
        ax.text(0.50, 0.35, f"Avg Response: {resp_t:.2f}s", ha="center", va="center", fontsize=7.5, color="#555555",
                transform=ax.transAxes)
                
        # Bottom Status Line: Realtime Request Outcomes (Option 1)
        rt_injected = self.shared_dict.get("rt_injected", 0)
        rt_processed = self.shared_dict.get("rt_processed", 0)
        rt_timed_out = self.shared_dict.get("rt_timed_out", 0)
        to_pct = (rt_timed_out / rt_injected * 100) if rt_injected > 0 else 0.0
        
        metrics_str = f"Injected: {rt_injected} | Processed: {rt_processed} | Timed Out: {rt_timed_out} ({to_pct:.1f}%)"
        ax.text(0.5, 0.12, metrics_str, ha="center", va="center", fontsize=7.5, color="#333333",
                bbox=dict(boxstyle="square,pad=0.25", facecolor="#f8f9fa", edgecolor="#ced4da", lw=0.8),
                transform=ax.transAxes)

    def _create_queue_length_plot(self, axis_id):
        ax = self._get_ax(axis_id)
        ax.set_title("Request Queue Backlog")
        ax.set_ylabel("Queue Length [req]")
        ax.set_xlabel("Simulation Time [h]")
        ax.plot(self._times, self._queue_lengths, color="darkorange", lw=1.8, label="Queue Backlog")
        ax.tick_params(axis="x", which="both", labelbottom=True)
        ax.tick_params(axis="y", which="both", labelleft=True)
        if self._queue_lengths:
            max_q = max(self._queue_lengths)
            ax.set_ylim(0, max(5, int(max_q * 1.2) + 1))

    def _create_realtime_lag_plot(self, axis_id):
        ax = self._get_ax(axis_id)
        ax.set_title("Real-Time Tick Duration & Lag")
        ax.set_ylabel("Duration [s]")
        ax.set_xlabel("Simulation Time [h]")
        
        if self._tick_durations and self._times:
            # Determine bar width dynamically based on time spacing
            if len(self._times) > 1:
                dt = (self._times[-1] - self._times[0]) / max(len(self._times) - 1, 1)
                bar_width = max(dt * 0.85, 0.0002)
            else:
                bar_width = 0.002

            # Color bars: Green = within budget, Yellow = exceeded step budget, Red = exceeded reopt budget
            bar_colors = []
            for i, dur in enumerate(self._tick_durations):
                step_b = self._step_budgets[i] if i < len(self._step_budgets) else 1.0
                reopt_b = self._reopt_budgets[i] if i < len(self._reopt_budgets) else step_b
                if dur > reopt_b:
                    bar_colors.append("#dc3545")  # Red: exceeded reopt budget
                elif dur > step_b:
                    bar_colors.append("#ffc107")  # Yellow: exceeded step budget
                else:
                    bar_colors.append("#28a745")  # Green: default / on time

            ax.bar(self._times, self._tick_durations, width=bar_width, color=bar_colors, align="center")

        # Budget reference lines
        if self._step_budgets:
            ax.plot(self._times, self._step_budgets, color="#e6b800", linestyle="--", lw=1.3, label="Step Budget")
        if self._reopt_budgets:
            ax.plot(self._times, self._reopt_budgets, color="#dc3545", linestyle="--", lw=1.3, label="Reopt Budget")

        ax.legend(loc="upper right", fontsize=7)
        ax.tick_params(axis="x", which="both", labelbottom=True)
        ax.tick_params(axis="y", which="both", labelleft=True)
        if self._tick_durations:
            max_t = max(max(self._tick_durations), max(self._reopt_budgets) if self._reopt_budgets else 1.0, max(self._step_budgets) if self._step_budgets else 1.0)
            ax.set_ylim(0, max(0.5, max_t * 1.25))

    def _create_queue_list_plot(self, axis_id):
        ax = self._get_ax(axis_id)
        ax.set_title("Request Queue Items", fontsize=12)
        
        queue_reqs = self.shared_dict.get("queue_requests", [])
        max_slots = max(5, len(queue_reqs))
        ax.set_ylim(-0.8, max_slots - 0.2)
        ax.set_yticks([])
        ax.tick_params(left=False, labelleft=False)
        ax.set_xlabel("Time Since Injection [s]")
        ax.tick_params(axis="x", which="both")

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#fd7e14", label="Waiting"),
            Patch(facecolor="#0d6efd", label="Processing"),
            Patch(facecolor="#6f42c1", label="Timed Out")
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=6.5, framealpha=0.85)

        if not queue_reqs:
            ax.text(0.5, 0.5, "Queue is empty\n(0 active requests)", ha="center", va="center", fontsize=9, color="#6c757d", transform=ax.transAxes)
            ax.set_xlim(0, 15)
            return

        elapsed = [r["elapsed_s"] for r in queue_reqs]
        states = [r["state"] for r in queue_reqs]

        # Color mapping:
        # waiting -> orange (#fd7e14)
        # processing -> blue (#0d6efd)
        # timed_out -> purple (#6f42c1)
        # processed -> green (#198754)
        color_map = {
            "waiting": "#fd7e14",
            "processing": "#0d6efd",
            "timed_out": "#6f42c1",
            "processed": "#198754"
        }
        bar_colors = [color_map.get(s, "#fd7e14") for s in states]

        y_pos = list(range(len(queue_reqs)))
        bars = ax.barh(y_pos, elapsed, color=bar_colors, height=0.6, align="center")

        # Label each bar with elapsed seconds and state tag
        for bar, el, state in zip(bars, elapsed, states):
            tag = "wait" if state == "waiting" else ("proc" if state == "processing" else ("timeout" if state == "timed_out" else "done"))
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f"{el:.1f}s [{tag}]", va="center", ha="left", fontsize=6.5, color="#333333")

        max_x = max(elapsed) if elapsed else 10
        ax.set_xlim(0, max(15, max_x * 1.35))
