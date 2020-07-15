"""Contains the random graph gen class"""

from flow.networks.base import Network
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from collections import defaultdict
import numpy as np
import re #RWH

ADDITIONAL_NET_PARAMS = {
    # dictionary of parameters to generate the graph
    # each edge will be directed
    "graph": {
        # number of nodes on the graph.
        "nodes": 5,
        # Size of the node m
        #'node_size' : 6,
        # probability of success of spawning an edge
        #"p": 0.5,
        # Initial position in tuple of the first node. Always deterministic. 
        #'initial' : [5,5],
        # lower or default between nodes 
        "lower_length": 100,
        # maximum length of the edges, if this is positive greater than 'lower length' then is chosen uniformly from both quantities
        #"max_length": -1,
    },
    # number of lanes per edge, as they are directed the number of lines follows the same direction
    "lanes": 1,
    # speed limit for all edges 
    "speed_limit": 35,
}


class RandomNodesNetwork(Network):
    """Random Graph network class.

    ROB WAS HERE.. and needs to be again to write this.

    Requires from net_params:

    * **grid_array** : dictionary of grid array data, with the following keys

      * **row_num** : number of horizontal rows of edges
      * **col_num** : number of vertical columns of edges
      * **inner_length** : length of inner edges in traffic light grid network
      * **short_length** : length of edges that vehicles start on
      * **long_length** : length of final edge in route
      * **cars_top** : number of cars starting at the edges heading to the top
      * **cars_bot** : number of cars starting at the edges heading to the
        bottom
      * **cars_left** : number of cars starting at the edges heading to the
        left
      * **cars_right** : number of cars starting at the edges heading to the
        right

    * **horizontal_lanes** : number of lanes in the horizontal edges
    * **vertical_lanes** : number of lanes in the vertical edges
    * **speed_limit** : speed limit for all edges. This may be represented as a
      float value, or a dictionary with separate values for vertical and
      horizontal lanes.

    Usage
    -----
    >>> from flow.core.params import NetParams
    >>> from flow.core.params import VehicleParams
    >>> from flow.core.params import InitialConfig
    >>> from flow.networks import TrafficLightGridNetwork
    >>>
    >>> network = RandomNodesNetwork(
    >>>     name='grid',
    >>>     vehicles=VehicleParams(),
    >>>     net_params=NetParams(
    >>>         additional_params={
    >>>             'grid_array': {
    >>>                 'row_num': 10,
    >>>                 'p': 0.9
    >>>                 'inner_length': 500,
    >>>                 'short_length': 1000,
    >>>             },
    >>>             'lanes' : 1,
    >>>             'speed_limit': 35
    >>>         },
    >>>     )
    >>> )
    """

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):

        """Initialize the n nodes network."""
        optional = ["tl_logic"]
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params and p not in optional:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        for p in ADDITIONAL_NET_PARAMS["graph"].keys():
            if p not in net_params.additional_params["graph"]:
                raise KeyError(
                    'graph parameter "{}" not supplied'.format(p))

        # retrieve all additional parameters
        # refer to the ADDITIONAL_NET_PARAMS dict for more documentation
        self.lanes = net_params.additional_params["lanes"]
        self.speed_limit = net_params.additional_params["speed_limit"]

        self.graph_params = net_params.additional_params["graph"]
        self.n = self.graph_params["nodes"]
        self.l = self.graph_params["lower_length"]

        # specifies whether or not there will be traffic lights at the
        # intersections (True by default)
        self.use_traffic_lights = net_params.additional_params.get(
            "traffic_lights", True)

        # radius of the inner nodes (ie of the intersections)
        #self.inner_nodes_radius = 2.9 + 3.3 * self.lanes

        # name of the network (DO NOT CHANGE)
        #self.name = "BobLoblawsLawBlog"
        self.name = "ThisIsROB"

        # RWH. A random seed that changes to generate some random paths
        self.rg = np.random.Generator(np.random.SFC64())

        # Starting the net generation
        self._start_

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)
        

    @property
    def _start_(self):
        def node_pos(node, n, l):
            col = int(node % n )
            row = int(node // n )
            return (col * l, row * l)

        def node_distances(node_id, nodes_pos, n):
            # Calculates the euclidian distance of all the nodes from the ref_node
            x_r, y_r = nodes_pos[node_id]
            distances = []
            sum_distances = 0
            for i in range(n):
                if i != node_id:
                    x, y = nodes_pos[i]
                    r = (x_r - x)**2 + (y_r - y)**2
                    d = np.sqrt(r)
                    sum_distances += d
                    distances.append(d)
                else:
                    distances.append([0])
            return distances, sum_distances

        # Chosing the random nodes
        nodes = self.rg.choice(range(self.n**2), size = self.n, replace=False)
        nodes_pos = [] # List to store the positions of the nodes on the net
        # Calculating their position on the grid
        for node in nodes:
            nodes_pos.append(node_pos(node, self.n, self.l))
        nodes_adj = [] # Adyacency list for the graph network
        nodes_deg = []
        nodes_calls = defaultdict(list) # Where the key is the node to go and the values the nodes from where
        # n^2 complexity
        for i in range(self.n):
            distances, sum_distances = node_distances(i, nodes_pos, self.n) # taking the i-th node as reference
            node_adj = []
            node_deg = 0
            for node_id, d_to_node in enumerate(distances):
                p_to_node = d_to_node / sum_distances
                if p_to_node >= self.rg.random(): # Event of success to connect node i to node_id
                    node_adj.append((node_id, p_to_node * sum_distances))
                    node_deg += 1 #outs
                    nodes_calls[node_id] += [i]
            nodes_deg.append(node_deg)
            nodes_adj.append(node_adj)
        # updating the deg with ins
        for i in nodes_calls.keys():
            nodes_deg[i] += len(nodes_calls[i])

        nodes_calls = nodes_calls # The keys are the node and the values the nodes that go to the key_node
        nodes_adj = nodes_adj # To generate the edges with _edges_
        # Format each i-th item for the i-th node, has a list with tuples
        # (node_to, distance_to)
        #self.nodes_deg = nodes_deg
        print(nodes_pos)
        print(nodes_adj)
        print(nodes_calls)
        print(nodes_deg)
        # *** Making nodes ***
        node_type = "traffic_light" if self.use_traffic_lights else "priority"

        nodes = []

        for i, pos in enumerate(nodes_pos):
            x, y = pos
            node_deg = nodes_deg[i]
            if node_deg > 1:
                nodes.append({
                    "id": "node{}".format(i),
                    "x": x,
                    "y": y,
                    "type": node_type,
                    "radius": 2.9 + 3.3 * self.lanes * nodes_deg[i]
                })
            elif node_deg == 1:
                nodes.append({"id": "node{}".format(i), "x": x, "y": y, "type": "priority"})
            else:
                #raise "Empty node. Sad news, something went wrong."
                None 

        self.nodes = nodes

        # *** Making edges ***
        self.edges = []
        self.routes = defaultdict(list) # For routes
        self.edge_starts = []
        for node_id, node_adj in enumerate(nodes_adj):
            for  to_node, to_distance in node_adj:
                edge = "edge{}_{}".format(node_id,to_node)
                self.edges.append({
                    "id": edge,
                    "priority": 78,
                    "type" : "normal_edge",
                    "from": "node{}".format(node_id),
                    "to": "node{}".format(to_node),
                    "length": to_distance
                })
                self.routes[edge] += [edge]
                self.edge_starts += [(edge, 1.0*to_node + 50.0*to_node)]

        # *** Making connections ***
        con_dict = {}

        def new_con(before_id, from_id, to_id, lane, signal_group):
            return [{
                "from": "edge{}_{}".format(before_id, from_id),
                "to": "edge{}_{}".format(from_id, to_id),
                "fromLane": str(lane),
                "toLane": str(lane),
                "signal_group": signal_group
            }]

        # As all the edges are directed, all the edges are connected in the nodes.
        # All possible connections are made
        for node_id, node_adj in enumerate(nodes_adj):
            conn = []
            for to_node, _ in node_adj:
                for before_node in nodes_calls[node_id]:
                    for i in range(self.lanes):
                        conn += new_con(before_node, node_id, to_node, i, 1)
            con_dict["node{}".format(node_id)] = conn
        
        self.con_dict = con_dict

        return "Generating graph. Done."
        

    def specify_nodes(self, net_params):
        """See parent class."""
        return self.nodes

    def specify_edges(self, net_params):
        """See parent class."""
        return self.edges

    def specify_routes(self, net_params):
        """See parent class."""
        return self.routes

    def specify_types(self, net_params):
        """See parent class."""
        types = [{
            "id": "normal_edge",
            "numLanes": self.lanes,
            "speed": self.speed_limit
        }]

        return types

    def specify_connections(self, net_params):
        """Build out connections at each inner node.

        Connections describe what happens at the intersections. Here we link
        lanes in straight lines, which means vehicles cannot turn at
        intersections, they can only continue in a straight line.
        """
        return self.con_dict

    # TODO necessary?
    def specify_edge_starts(self):
        """See parent class."""
        return self.edge_starts