from flow.controllers import IDMController, ContinuousRouter
from flow.core.params import SumoParams, EnvParams,VehicleParams, \
    InitialConfig, NetParams, TrafficLightParams, \
    InFlows, SumoCarFollowingParams
from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.networks.traffic_light_grid import TrafficLightGridNetwork
from flow.core.experiment import Experiment

### Traffic ligths parameters
tl_type = "actuated"
program_id = 1
max_gap = 3.0
detector_gap = 0.8
show_detectors = True
phases = [{"duration": "31", "minDur": "8", "maxDur": "45", "state": "GrGr"},
        {"duration": "3", "minDur": "3", "maxDur": "6", "state": "yryr"},
        {"duration": "5", "minDur": "3", "maxDur": "10", "state": "rGrG"},
        {"duration": "3", "minDur": "3", "maxDur": "6", "state": "ryry"}]
tl_logic = TrafficLightParams(baseline=True)
tl_nodes = range(1,7)
for node in tl_nodes:
    tl_logic.add(
                "center{}".format(node), 
                tls_type=tl_type, 
                programID=program_id, 
                phases=phases, 
                maxGap=max_gap, 
                detectorGap=detector_gap, 
                showDetectors=show_detectors
    )
### Network Parameters
inner_length = 200
long_length = 300
short_length = 300
n = 2 # rows
m = 5 # columns
num_cars_left = 5
num_cars_right = 5
num_cars_top = 5
num_cars_bot = 5
tot_cars = (num_cars_left + num_cars_right) * m \
    + (num_cars_top + num_cars_bot) * n

grid_array = {"short_length": short_length, "inner_length": inner_length,
              "long_length": long_length, "row_num": n, "col_num": m,
              "cars_left": num_cars_left, "cars_right": num_cars_right,
              "cars_top": num_cars_top, "cars_bot": num_cars_bot}
additional_net_params = {"grid_array": grid_array, "speed_limit": 35,
                         "horizontal_lanes": 1, "vertical_lanes": 1,
                         "traffic_lights": True, "tl_logic": tl_logic}

net_params = NetParams(additional_params=additional_net_params)
### Vehicle Parameters
vehicles = VehicleParams()
vehicles.add(veh_id="human",
             acceleration_controller=(IDMController, {}),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=tot_cars)
### Simulator parameters
sim_params = SumoParams(sim_step=0.1, render='rgb', show_radius=True,)
initial_config = InitialConfig(spacing="custom", bunching=15)
### Environment parameters
env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS, evaluate=True,
    horizon=2000)
### Final: Flow parameters
flow_params = dict(
    exp_tag="TrafficLigth",
    env_name=AccelEnv,
    network= TrafficLightGridNetwork,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
)
exp = Experiment(flow_params)
# run the sumo simulation
_ = exp.run(1)