from flow.core.params import VehicleParams
from flow.controllers import IDMController, ContinuousRouter
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.networks.minicity import MiniCityNetwork
from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.networks.random_graph import ADDITIONAL_NET_PARAMS, RandomGridNetwork
from flow.core.experiment import Experiment

NUM_VEHICLES = 100
vehicles = VehicleParams()
vehicles.add(veh_id="human",
             acceleration_controller=(IDMController, {}),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=NUM_VEHICLES)
             
sim_params = SumoParams(sim_step=0.1, render=True)
initial_config = InitialConfig(spacing="uniform", bunching=15)
env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

additional_net_params = ADDITIONAL_NET_PARAMS.copy()
additional_net_params["grid"]["h_grid"] = 10
additional_net_params["grid"]["w_grid"] = 5
additional_net_params["grid"]["long_length"] = 200
additional_net_params["lanes"] = 2
additional_net_params["speed_limit"] = 70
additional_net_params['type_of_cell'] = [1,2,3,4,5,6]
net_params = NetParams(additional_params=additional_net_params)

flow_params = dict(
    exp_tag="Random",
    env_name=AccelEnv,
    network= RandomGridNetwork,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
)
# number of time steps
flow_params['env'].horizon = 1000
exp = Experiment(flow_params)
# run the sumo simulation
_ = exp.run(1)