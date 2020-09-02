# the TestEnv environment is used to simply simulate the network
from flow.envs import TestEnv

# the Experiment class is used for running simulations
from flow.core.experiment import Experiment

# all other imports are standard
from flow.core.params import VehicleParams
from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import EnvParams
from flow.core.params import SumoParams
from flow.networks import Network
from multiprocessing import Pool

net_params = NetParams(
    osm_path='/home/roberto/flow/tutorials/networks/bay_bridge.osm'
) # Absulute path

# create the remainding parameters
env_params = EnvParams()
sim_params = SumoParams(render=False)
initial_config = InitialConfig()
vehicles = VehicleParams()
vehicles.add('human', num_vehicles=100)

flow_params = dict(
    exp_tag='bay_bridge',
    env_name=TestEnv,
    network=Network,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
)


def run_exp(flow_params, horizon=1000):
    flow_params['env'].horizon = horizon
    exp = Experiment(flow_params)
    _ = exp.run(1)

NUM_ITERS = 10

with Pool(processes=4) as P:
    results = P.imap_unordered(run_exp, [flow_params for _ in range(NUM_ITERS)])
    P.close()
    P.join()
print(results)
# IT WORKS!