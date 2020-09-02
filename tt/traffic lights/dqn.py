"""
File to set and train a DQN arquitecture in a network with traffic ligth agents
with human controlled vehicles.

@author: bobobert
"""
# ESCENTIALS IMPORTS
import os
import random
import time
# FLOW IMPORTS
from flow.envs.multiagent import MultiTrafficLightGridPOEnv
from flow.networks import TrafficLightGridNetwork
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import InFlows, SumoCarFollowingParams, VehicleParams
from flow.controllers import SimCarFollowingController, GridRouter, GridRandomRouter
from flow.core.experiment import Experiment
from flow.utils.registry import make_create_env

# Experiment parameters
N_ROLLOUTS = 100  # number of rollouts per training iteration
N_CPUS = os.cpu_count() - 1  # number of parallel workers

# Environment parameters
MAX_SPEED = int(60 / 3.6)  # enter speed for departing vehicles for the generation in m/s
INNER_LENGTH = 300  # length of inner edges in the traffic light grid network
LONG_LENGTH = 100  # length of final edge in route
SHORT_LENGTH = 300  # length of edges that vehicles start on
# number of vehicles originating in the left, right, top, and bottom edges
N_LEFT, N_RIGHT, N_TOP, N_BOTTOM = 1, 1, 1, 1

EDGE_INFLOW = 300  # inflow rate of vehicles per hour at every edge
N_ROWS = 3  # number of row of bidirectional lanes
N_COLUMNS = 3  # number of columns of bidirectional lanes
N_LANES = 3

# we place a sufficient number of vehicles to ensure they confirm with the
# total number specified above. We also use a "right_of_way" speed mode to
# support traffic light compliance
N_VEHICLES = (N_LEFT + N_RIGHT) * N_COLUMNS + (N_BOTTOM + N_TOP) * N_ROWS
vehicles = VehicleParams()

#SIMULATOR OPTIONS
T_STEP = 0.25 # Time per step in the simulations in seconds
HORIZON = 400  # time horizon in seconds of a single rollout
RENDER = False
WARNINGS = False
EVALUATE = True
WARMUP_STEPS = random.randint(0,100)

vehicles.add(
    veh_id="human",
    acceleration_controller=(SimCarFollowingController, {}),
    car_following_params=SumoCarFollowingParams(
        min_gap=2.5,
        max_speed=MAX_SPEED,
        decel=8.0,  # avoid collisions at emergency stops
        speed_mode="right_of_way",
    ),
    routing_controller=(GridRandomRouter, {}),
    num_vehicles=N_VEHICLES)

# inflows of vehicles are place on all outer edges (listed here)
outer_edges = []
outer_edges += ["left{}_{}".format(N_ROWS, i) for i in range(N_COLUMNS)]
outer_edges += ["right0_{}".format(i) for i in range(N_ROWS)]
outer_edges += ["bot{}_0".format(i) for i in range(N_ROWS)]
outer_edges += ["top{}_{}".format(i, N_COLUMNS) for i in range(N_ROWS)]

# equal inflows for each edge (as dictate by the EDGE_INFLOW constant)
inflow = InFlows()
for edge in outer_edges:
    inflow.add(
        veh_type="human",
        edge=edge,
        vehs_per_hour=EDGE_INFLOW,
        depart_lane="free",
        depart_speed=MAX_SPEED)
# Setting the experiment parameters

flow_params = dict(
    exp_tag = "grid_0_{}x{}_i{}_multiagent".format(N_ROWS, N_COLUMNS, EDGE_INFLOW),

    env_name = MultiTrafficLightGridPOEnv,

    network = TrafficLightGridNetwork,

    simulator = 'traci',

    sim = SumoParams(
        restart_instance=True,
        sim_step=T_STEP,
        render=RENDER,
        print_warnings=WARNINGS,
        
    ),
    env = EnvParams(
        warmup_steps=WARMUP_STEPS,
        horizon=HORIZON,
        evaluate=EVALUATE,
        additional_params={
            "target_velocity": 50,
            "switch_time": 3,
            "num_observed": 4, #Number of vehicles observed per edge in the network
            "discrete": False,
            "tl_type": "actuated",
            "num_local_edges": 4, # DEFAULT, not in use
            "num_local_lights": 4,
        },
    ),
    net = NetParams(
        inflows=inflow,
        reroute=True,
        additional_params={
            "speed_limit": MAX_SPEED + 5,  # inherited from grid0 benchmark
            "grid_array": {
                "short_length": SHORT_LENGTH,
                "inner_length": INNER_LENGTH,
                "long_length": LONG_LENGTH,
                "row_num": N_ROWS,
                "col_num": N_COLUMNS,
                "cars_left": N_LEFT,
                "cars_right": N_RIGHT,
                "cars_top": N_TOP,
                "cars_bot": N_BOTTOM,
            },
            "horizontal_lanes": N_LANES,
            "vertical_lanes": N_LANES,
        },
    ),
    veh = vehicles,

    initial = InitialConfig(
        spacing='custom',
        shuffle=True,
    ),
)

# Creating the environment
create_env, env_name = make_create_env(params=flow_params, version=0)
env = create_env()
action_space = env.action_space

# LINES FOR TESTING THE ENV
traffic_lights = env.k.traffic_light.get_ids()
def arm_rl():
    rl_actions = dict()
    for id in traffic_lights:
        rl_actions[id] = action_space.sample()
    return rl_actions
env.reset()
#print("Phase for center0", env.k.traffic_light.get_state('center0'))
start_sim = time.time()
for _ in range(int(HORIZON / T_STEP)):
    step_rl = arm_rl()
    env.step(rl_actions = step_rl)
    state = env.get_state()
    reward = env.compute_reward(step_rl)
for key,val in state.items():
    print("For {} a list with {} elements, reward {}".format(key, len(val), reward[key]), val)
run_time = time.time() - start_sim
print("Total time {} m {} s for simulation to end".format(run_time//60, (run_time%60)//1))