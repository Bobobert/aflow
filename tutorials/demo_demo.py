
# import Flow's base network class
from flow.networks import Network

# define the network class, and inherit properties from the base network class
class myNetwork(Network):
    pass

ADDITIONAL_NET_PARAMS = {
    "radius": 100,
    "num_lanes": 2,
    "speed_limit": 60,
}

class myNetwork(myNetwork):  # update my network class

    def specify_nodes(self, net_params):
        # one of the elements net_params will need is a "radius" value
        r = net_params.additional_params["radius"]

        # specify the name and position (x,y) of each node
        nodes = [{"id": "bottom", "x": 0,  "y": -r},
                 {"id": "right",  "x": r,  "y": 0},
                 {"id": "top",    "x": 0,  "y": r},
                 {"id": "left",   "x": -r, "y": 0}]

        return nodes

# some mathematical operations that may be used
from numpy import pi, sin, cos, linspace

class myNetwork(myNetwork):  # update my network class

    def specify_edges(self, net_params):
        r = net_params.additional_params["radius"]
        edgelen = r * pi / 2
        # this will let us control the number of lanes in the network
        lanes = net_params.additional_params["num_lanes"]
        # speed limit of vehicles in the network
        speed_limit = net_params.additional_params["speed_limit"]

        edges = [
            {
                "id": "edge0",
                "numLanes": lanes,
                "speed": speed_limit,     
                "from": "bottom", 
                "to": "right", 
                "length": edgelen,
                "shape": [(r*cos(t), r*sin(t)) for t in linspace(-pi/2, 0, 40)]
            },
            {
                "id": "edge1",
                "numLanes": lanes, 
                "speed": speed_limit,
                "from": "right",
                "to": "top",
                "length": edgelen,
                "shape": [(r*cos(t), r*sin(t)) for t in linspace(0, pi/2, 40)]
            },
            {
                "id": "edge2",
                "numLanes": lanes,
                "speed": speed_limit,
                "from": "top",
                "to": "left", 
                "length": edgelen,
                "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi/2, pi, 40)]},
            {
                "id": "edge3", 
                "numLanes": lanes, 
                "speed": speed_limit,
                "from": "left", 
                "to": "bottom", 
                "length": edgelen,
                "shape": [(r*cos(t), r*sin(t)) for t in linspace(pi, 3*pi/2, 40)]
            }
        ]

        return edges

from flow.core.params import VehicleParams
from flow.controllers import IDMController, ContinuousRouter
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.networks.minicity import MiniCityNetwork

vehicles = VehicleParams()
vehicles.add(veh_id="human",
             acceleration_controller=(IDMController, {}),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=22)

sim_params = SumoParams(sim_step=5, render=True)

initial_config = InitialConfig(bunching=40)

from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS

env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

additional_net_params = ADDITIONAL_NET_PARAMS.copy()
net_params = NetParams(additional_params=additional_net_params)
City = MiniCityNetwork('City', vehicles, net_params)


from flow.core.experiment import Experiment

flow_params = dict(
    exp_tag='test_network',
    env_name=AccelEnv,
    network= myNetwork,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
)

# number of time steps
flow_params['env'].horizon = 2000
exp = Experiment(flow_params)

# run the sumo simulation
_ = exp.run(1)