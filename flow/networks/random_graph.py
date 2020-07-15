"""Contains the random graph gen class"""

from flow.networks.base import Network
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from collections import defaultdict
import numpy as np

ADDITIONAL_NET_PARAMS = {
    # dictionary of parameters to generate the graph
    # each edge will be directed
    "grid": {
        # Heigth of the grid.
        "h_grid": 5,
        # Width of the grid
        "w_grid": 5,
        # Length of the square cells in m
        "lower_length": 100,
        # Initial position of top-left the corner of the grid
        'initial_pos': (5,5),
        # Curve resolution
        'resolution': 15,
    },
    # number of lanes per edge, as they are directed the number of lines follows the same direction
    "lanes": 1,
    # speed limit for all edges 
    "speed_limit": 35,
    # Type of cells that will be available to the network generation
    "type_of_cell": [0, 1, 2, 3, 4, 5, 6]
}


class RandomGridNetwork(Network):
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

        for p in ADDITIONAL_NET_PARAMS["grid"].keys():
            if p not in net_params.additional_params["grid"]:
                raise KeyError(
                    'grid parameter "{}" not supplied'.format(p))

        # retrieve all additional parameters
        # refer to the ADDITIONAL_NET_PARAMS dict for more documentation
        self.lanes = net_params.additional_params["lanes"]
        self.speed_limit = net_params.additional_params["speed_limit"]

        self.grid_params = net_params.additional_params["grid"]
        self.n_col = self.grid_params["w_grid"]
        self.n_row = self.grid_params["h_grid"]
        self.l = self.grid_params["lower_length"] / 2
        self.initial_pos = self.grid_params['initial_pos']
        self.resolution = self.grid_params['resolution']
        self.type_of_cell = net_params.additional_params['type_of_cell']

        # specifies whether or not there will be traffic lights at the
        # intersections (True by default)
        self.use_traffic_lights = net_params.additional_params.get(
            "traffic_lights", True)

        # radius of the inner nodes (ie of the intersections)
        #self.inner_nodes_radius = 2.9 + 3.3 * self.lanes

        # name of the network (DO NOT CHANGE)
        #self.name = "BobLoblawsLawBlog"
        self.name = "ThisIsROB"

        # RWH. A random seed that changes to generate the random choices
        self.rg = np.random.Generator(np.random.SFC64())

        # Starting the net generation
        self._start_
        for node in self.nodes:
            print(node['id'], node['x'], node['y'])
        for edge in self.edges:
            try:
                print(edge['id'], edge['shape'])
            except:
                None
        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)
        

    @property
    def _start_(self):
        """
        Each space on the grid in compose of a cell.
        The cells have by default 4 nodes in them with an optional center on them.
                
                 ---x---
                |       |
                x   o   x
                |       |
                 ---x---

        When generating the grid, each cell is chosen by the random generator.
        The type of cell that can happen is indicated by an index as follows
        0: Empty
         ---x---
        |       |
        x       x
        |       |
         ---x---
         1: Start/dead end
         ---x---
        |       |
        x   o   x
        |   |   |
         ---x---
         2: Straight paht
         ---x---
        |   |   |
        x   o   x
        |   |   |
         ---x---
         3: T intersection
         ---x---
        |   |   |
        x   o---x
        |   |   |
         ---x---
         4: Crossroad
         ---x---
        |   |   |
        x---o---x
        |   |   |
         ---x---
         5: Single Turn
         ---x---
        |       |
        x   .---x
        |   |   |
         ---x---
         6: double turn (Cannot go straigth ways)
         ---x---
        |   |   |
        x---.---x
        |   |   |
         ---x---

        After a type of cell has been chosen, The rotation in 90° is made with 
        uniform distribution.
        0: 0°
        1: 90°
        2: 180°
        3: 270°
        """
        #type_of_cell = [0, 1, 3, 4, 5, 6]
        cells_nodes = {
            0: ([0,0,0,0], False),
            1: ([0,0,0,1], True),
            2: ([0,1,0,1], True),
            3: ([1,1,0,1], True),
            4: ([1,1,1,1], True),
            5: ([1,0,0,1], False),
            6: ([1,1,1,1], False),
            7: ([1,0,0,1], True)
        }
        cells_edges = {
            0: [0,0,0,0],
            1: [0,0,0,1],
            2: [0,1,0,1],
            3: [1,1,0,1],
            4: [1,1,1,1],
            5: [1,0,0,1],
            6: [1,1,1,1],
            7: [1,0,0,1],
        }
        rotations = [0, 1, 2, 3]
        # Generating the grid
        grid_cells = self.rg.choice(self.type_of_cell,\
                                    size=self.n_col*self.n_row,\
                                    replace=True)
        grid_cells = grid_cells.reshape((self.n_row, self.n_col))
        grid_rotations = self.rg.choice(rotations,\
                                    size=self.n_col*self.n_row,\
                                    replace=True)
        grid_rotations = grid_rotations.reshape((self.n_row, self.n_col))
        # It has the by 3 multiplier to have the cells as they requiere
        grid_nodes = np.zeros((2*self.n_row + 1, 2*self.n_col + 1), dtype=np.bool_)
        grid_edges = np.zeros((self.n_row, self.n_col, 4), dtype=np.bool_)

        def apply_rotation(cell, rotation):
            nodes, center = cells_nodes[cell]
            edges = cells_edges[cell]
            # Making the lists of nodes/edges to shift/rotate
            nodes = nodes[rotation:] + nodes[:rotation] 
            edges = edges[rotation:] + edges[:rotation]
            grid_pad = np.zeros((3,3), dtype=np.bool_)
            grid_pad[1,2] = nodes[0]
            grid_pad[0,1] = nodes[1]
            grid_pad[1,0] = nodes[2]
            grid_pad[2,1] = nodes[3]
            if center:
                grid_pad[1,1] = 1
            return grid_pad, edges
            
        for i in range(self.n_row):
            for j in range(self.n_col):
                cell = grid_cells[i,j]
                rotation = grid_rotations[i,j]
                nodes, edges = apply_rotation(cell, rotation)
                grid_nodes[2*i:2*i+3,2*j:2*j+3] += nodes
                grid_edges[i,j] = edges

        self.nodes = []
        # Definining the nodes
        for i in range(2*self.n_row +1):
            for j in range(2*self.n_col +1):
                node = grid_nodes[i,j]
                if (i % 2 == 1) and (j % 2 == 1) and node:
                    # Center detected
                    self.nodes += [{
                        "id": "node{}".format(j+i*(2*self.n_col+1)),
                        "x": i*self.l + self.initial_pos[0],
                        "y": j*self.l + self.initial_pos[1],
                        "type": "traffic_light" if self.use_traffic_lights else "priority",
                        "radius": 5 + 3.3 * self.lanes * 2,
                    }]
                elif node:
                    self.nodes += [{
                        "id": "node{}".format(j+i*(2*self.n_col+1)),
                        "x": i*self.l + self.initial_pos[0],
                        "y": j*self.l + self.initial_pos[1],
                        "type": "priority",
                        "radius": 5 + 3.3 * self.lanes * 2,
                    }]

        self.edges = []
        #self.edge_starts = []
        def new_edge(from_node, to_node, pos=None, rotation=0):
            pi = np.pi
            if isinstance(pos, tuple):
                ranges = [
                    (pi*1.5, 2*pi,   1, 1, -1, 1),
                    (0, pi*0.5,      1, 0, -1, 1),
                    (0.5*pi, 1*pi,   0, 0, -1, 1),
                    (pi, 1.5*pi,     0, 1, 1, -1)
                ]
                rn = ranges[rotation]
                r = self.l
                
                shape = [(
                        r * np.cos(t) * rn[4]  + (rn[2] + pos[0]) * 2 * r + self.initial_pos[0],
                        r * np.sin(t) * rn[5] + (rn[3] + pos[1]) * 2 * r + self.initial_pos[1])
                        for t in np.linspace(rn[0], rn[1], self.resolution)
                        ]

                return [{
                        "id": "edge{}_{}".format(from_node, to_node),
                        "priority": 78,
                        "type" : "normal_edge",
                        "from": "node{}".format(from_node),
                        "to": "node{}".format(to_node),
                        "length": self.l*pi/2,
                        "shape": shape
                    },{
                        "id": "edge{}_{}".format(to_node, from_node),
                        "priority": 78,
                        "type" : "normal_edge",
                        "from": "node{}".format(to_node),
                        "to": "node{}".format(from_node),
                        "length": self.l*pi/2,
                        "shape": shape
                    }]
            else:
                return [{
                        "id": "edge{}_{}".format(from_node, to_node),
                        "priority": 78,
                        "type" : "normal_edge",
                        "from": "node{}".format(from_node),
                        "to": "node{}".format(to_node),
                        "length": self.l
                    },{
                        "id": "edge{}_{}".format(to_node, from_node),
                        "priority": 78,
                        "type" : "normal_edge",
                        "from": "node{}".format(to_node),
                        "to": "node{}".format(from_node),
                        "length": self.l
                    }]
                    
        def resolve_rotation(edges):
            resolved = []
            for k,edge in enumerate(edges):
                if edge:
                    if k == 0:
                        resolved.append((2*j+2)+(2*i+1)*(self.n_col*2+1))
                    elif k == 1:
                        resolved.append((2*j+1)+(2*i)*(self.n_col*2+1))
                    elif k == 2:
                        resolved.append((2*j)+(2*i+1)*(self.n_col*2+1))
                    elif k == 3:
                        resolved.append((2*j+1)+(2*i+2)*(self.n_col*2+1))
            return resolved

        con_dict = {} # Yet to be implemented. Not needed tho
        def new_con(before_id, from_id, to_id, lane, signal_group):
            return [{
                "from": "edge{}_{}".format(before_id, from_id),
                "to": "edge{}_{}".format(from_id, to_id),
                "fromLane": str(lane),
                "toLane": str(lane),
                "signal_group": signal_group
            }]

        # Defining edges
        for i in range(self.n_row):
            for j in range(self.n_col):
                resolved_nodes = resolve_rotation(grid_edges[i,j])
                rotation = grid_rotations[i,j]
                cell_type = grid_cells[i,j]
                if cell_type == 5: # Cicle type of cell
                    self.edges += new_edge(resolved_nodes[0], resolved_nodes[1], pos=(i,j), rotation=rotation)
                elif cell_type == 6: # Circle type of cell
                    assert len(resolved_nodes) == 4, "Not good. Something bad happened."
                    if rotation % 2 == 1: # A non rotation happened, 1 or 3 are the same result
                        self.edges += new_edge(resolved_nodes[0], resolved_nodes[1], pos=(i,j), rotation=1)
                        self.edges += new_edge(resolved_nodes[2], resolved_nodes[3], pos=(i,j), rotation=3)
                    else: # An even rotation happened
                        self.edges += new_edge(resolved_nodes[0], resolved_nodes[3], pos=(i,j), rotation=0)
                        self.edges += new_edge(resolved_nodes[1], resolved_nodes[2], pos=(i,j), rotation=2)
                else: # A normal cell kind
                    for to_node in resolved_nodes:
                        # If exists
                        from_node = ((2*j+1)+(2*i+1)*(self.n_col*2+1)) #Center node
                        self.edges += new_edge(from_node, to_node)


    def specify_nodes(self, net_params):
        """See parent class."""
        return self.nodes

    def specify_edges(self, net_params):
        """See parent class."""
        return self.edges

    def specify_types(self, net_params):
        """See parent class."""
        types = [{
            "id": "normal_edge",
            "numLanes": self.lanes,
            "speed": self.speed_limit
        }]
        return types

    def specify_edge_starts(self):
        """See parent class"""
        #return self.edge_starts
        return None

    @staticmethod # Not implemented yet
    def gen_custom_start_pos(cls, net_params, initial_config, num_vehicles):
        """See parent class."""
        """grid = net_params.additional_params["grid"]
        row_num = grid["h_grid"]
        col_num = grid["w_grid"]

        start_pos = []

        x0 = 5  # position of the first car
        dx = initial_config.bunching  # distance between each car

        start_lanes = []
        for i in range(col_num):
            start_pos += [("right0_{}".format(i), x0 + k * dx)
                          for k in range(cars_heading_right)]
            start_pos += [("left{}_{}".format(row_num, i), x0 + k * dx)
                          for k in range(cars_heading_left)]
            horz_lanes = np.random.randint(low=0, high=net_params.additional_params["horizontal_lanes"],
                                           size=cars_heading_left + cars_heading_right).tolist()
            start_lanes += horz_lanes

        for i in range(row_num):
            start_pos += [("top{}_{}".format(i, col_num), x0 + k * dx)
                          for k in range(cars_heading_top)]
            start_pos += [("bot{}_0".format(i), x0 + k * dx)
                          for k in range(cars_heading_bot)]
            vert_lanes = np.random.randint(low=0, high=net_params.additional_params["vertical_lanes"],
                                           size=cars_heading_left + cars_heading_right).tolist()
            start_lanes += vert_lanes

        return start_pos, start_lanes"""
        None

 