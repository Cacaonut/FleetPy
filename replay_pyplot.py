import os
import sys
import argparse

from src.ReplayFromResult import ReplayPyPlot


def main(output_dir, sim_seconds_per_real_second, start_time_in_seconds = None, end_time_in_seconds = None,
          plot_extend = None, live_plot = True, create_images = True,parcels = False,passengers = False,
          map_plot = "vehicle_status", plot_1=None, plot_2=None, plot_3=None,
          plot_4=None, plot_5=None, plot_6=None,
          color_list = False,plot_args=[1,1,1,0,0,0]):
    """This function uses pyplot to visualize the fleet operation.

    :param output_dir: path to result directory
    :param sim_seconds_per_real_second: determines the speed of the replay
    :param start_time_in_seconds: determines simulation time when replay is started
    :param end_time_in_seconds: determines simulation time when replay ends
    :param plot_extent: Tuple of (lon_1, lon_2, lat_1, lat_2) marking the left bottom and top right boundary of
                        the map (EPSG_WGS = 4326); if None -> extend of loaded network chosen
    :param live_plot: if True: plots directly shown; else: figures stored in ouputdir/plots    :param parcels: if True: plots parcel data
    :param passengers: if True: plots passenger data
    :param map_plot: options: "vehicle_status", "occupancy"
    :param plot_1: options are (str): status_count, occupancy_count, occupancy_average, occupancy_stack_chart, waiting_time_average, ride_time_average, detour_time_average, service_rate, realtime_architecture, queue_length, realtime_lag
    :param plot_2: options are (str): same as plot_1
    :param plot_3: options are (str): same as plot_1
    :param plot_4: column 2 top plot (str): same as plot_1
    :param plot_5: column 2 mid plot (str): same as plot_1
    :param plot_6: column 2 bottom plot (str): same as plot_1
    :param color_list: if True: color list is used for plotting
    :param plot_args: list of integers that determine which plots are created
    
    :return:
    """
    if not os.path.isdir(output_dir):
        raise IOError(f"Result directory {output_dir} not found!")
    sim_seconds_per_real_second = float(sim_seconds_per_real_second)
    replay = ReplayPyPlot(live_plot=live_plot, create_images = create_images,
                          parcels = parcels,passengers = passengers,
                          map_plot=map_plot, plot_1=plot_1, plot_2=plot_2, plot_3=plot_3,
                          plot_4=plot_4, plot_5=plot_5, plot_6=plot_6,
                          plot_args = plot_args,
                          color_list = color_list)
    if start_time_in_seconds is not None and end_time_in_seconds is not None:
        replay.load_scenario(output_dir, start_time_in_seconds=int(start_time_in_seconds)
                             , end_time_in_seconds = int(end_time_in_seconds), plot_extend=plot_extend)
    elif start_time_in_seconds is not None:
        replay.load_scenario(output_dir, start_time_in_seconds=int(start_time_in_seconds)
                             , plot_extend=plot_extend)
    elif end_time_in_seconds is not None:
        replay.load_scenario(output_dir, end_time_in_seconds=int(end_time_in_seconds)
                             , plot_extend=plot_extend)
    else:
        replay.load_scenario(output_dir, plot_extend=plot_extend)
    replay.set_time_step(sim_seconds_per_real_second)
    replay.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Replay a simulation result.')
    parser.add_argument('output_dir', type=str, help='path to result directory')
    parser.add_argument('sim_seconds_per_real_second', type=float, help='determines the speed of the replay')
    parser.add_argument('--start_time_in_seconds', type=int, help='determines simulation time when replay is started')
    parser.add_argument('--end_time_in_seconds', type=int, help='determines simulation time when replay ends')
    parser.add_argument('--save_figs', action='store_true', help='if set, figures are stored in ouputdir/plots. Otherwise, plots are shown live')
    AVAILABLE_PLOTS_HELP = "status_count, occupancy_count, occupancy_average, occupancy_stack_chart, waiting_time_average, ride_time_average, detour_time_average, service_rate, realtime_architecture, queue_length, realtime_lag, queue_list"
    parser.add_argument('--plot_1', type=str, default=None, help=f'determines plot 1 (col 1 top): {AVAILABLE_PLOTS_HELP}')
    parser.add_argument('--plot_2', type=str, default=None, help=f'determines plot 2 (col 1 mid): {AVAILABLE_PLOTS_HELP}')
    parser.add_argument('--plot_3', type=str, default=None, help=f'determines plot 3 (col 1 bottom): {AVAILABLE_PLOTS_HELP}')
    parser.add_argument('--plot_4', type=str, default=None, help=f'determines plot 4 (col 2 top): {AVAILABLE_PLOTS_HELP}')
    parser.add_argument('--plot_5', type=str, default=None, help=f'determines plot 5 (col 2 mid): {AVAILABLE_PLOTS_HELP}')
    parser.add_argument('--plot_6', type=str, default=None, help=f'determines plot 6 (col 2 bottom): {AVAILABLE_PLOTS_HELP}')
    # add argument called status_type that takes two values passenger and parcel
    parser.add_argument('--map_plot', type=str, 
                        help='determines the type of status to plot Either "occupancy" or "vehicle_status" or "zone"',default="occupancy")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--parcels',action='store_true',help='if set, plots parcel data',default=False)
    group.add_argument('--passengers',action='store_true',help='if set, plots passenger data',default=False)

    args = parser.parse_args()

    if args.map_plot != "occupancy" and args.map_plot != "vehicle_status" and args.map_plot != "zone":
        raise IOError("Incorrect map plot type!")

    main(args.output_dir, args.sim_seconds_per_real_second, start_time_in_seconds = args.start_time_in_seconds,
         end_time_in_seconds = args.end_time_in_seconds, parcels = args.parcels, passengers = args.passengers,
         map_plot = args.map_plot, plot_1=args.plot_1, plot_2=args.plot_2, plot_3=args.plot_3,
         plot_4=args.plot_4, plot_5=args.plot_5, plot_6=args.plot_6,
         live_plot=not args.save_figs,color_list = False)
    



