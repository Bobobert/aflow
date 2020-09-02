"""Environment used to train vehicles to improve traffic on a highway."""
import numpy as np
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces.tuple import Tuple
from flow.core.rewards import desired_velocity
from flow.envs.multiagent.base import MultiEnv

# Aditional paramaters for this environment
ADDITIONAL_ENV_PARAMS = {
    # Radius of nearby automoviles
    "radius": 50, 
    # Maximum number of cars connected in a radius
    "max_connections": 128,
    # maximum acceleration of autonomous vehicles
    'max_accel': 1,
    # maximum deceleration of autonomous vehicles
    'max_decel': 1,
    # Maximum velocity allowed in the grid in m/s
    'max_vel' : 120/3.6,
    # Maximum number of actions
    'max_act' : 3 # 0 - keep straigth, 1 - turn left, 2 - turn rigth.
}

class V2V(MultiEnv):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        super().__init__(env_params, sim_params, network, simulator)

        self.R = env_params.additional_params.get('radius')
        self.max_connections = env_params.additional_params.get('max_connections')
    
    @abstractmethod
    def _apply_rl_actions(self, rl_actions):
        pass

    @abstractmethod
    def get_state(self):
        """Return the state of the simulation as perceived by the RL agent.

        The state of the agent is done by extrating the velocity and aceleration present
        in the simulator, and the positions and relative velocities from all the agents 
        around the agent. If this number exceeds the max_connections it chooses the nearest
        agents with the euclidean distance

        Returns
        -------
        state : array_like
            information on the state of the vehicles, which is provided to the
            agent
        """
        pass

    @property
    @abstractmethod
    def action_space(self):
        """Identify the dimensions and bounds of the action space.

        Action space is just to control de aceleration of the vehicle and its
        movement control. 

        Returns
        -------
        gym Box or Tuple type
            a bounded box depicting the shape and bounds of the action space
        """
        acel_space = Box(
            low=-np.abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(1,),
            dtype=np.float32)
        mov_space = Discrete(self.env_params.additional_params['max_act'])

        return Tuple((acel_space, mov_space))
         

    @property
    @abstractmethod
    def observation_space(self):
        """Identify the dimensions and bounds of the observation space.

        It has the aceleration and velocity of its own and the positions and velocities of 
        the nearest vehicles around it.

        Returns
        -------
        gym Box or Tuple type
            a bounded box depicting the shape and bounds of the observation
            space
        """
        acel_space = Box(
            low=-np.abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(1,),
            dtype=np.float32)
        vel_space = Box(
            low=0,
            high=self.env_params.additional_params['max_vel'],
            shape=(1,),
            dtype=np.float32)
        near_cars_pos = Box(
            low=-self.R,
            high=self.R,
            shape=(self.max_connections,2),
            dtype=np.float32)
        )
        near_cars_vel = Box(
            low=-float('inf'),
            high=float('inf'),
            shape(self.max_connections),
            dtype=np.float32
        )
        
        return Tuple((acel_space, 
                        vel_space, 
                        mov_space,
                        near_cars_pos,
                        near_cars_vel))
        
    def compute_reward(self, rl_actions, **kwargs):
        """Reward function for the RL agent(s).

        MUST BE implemented in new environments.
        Defaults to 0 for non-implemented environments.

        Parameters
        ----------
        rl_actions : array_like
            actions performed by rl vehicles
        kwargs : dict
            other parameters of interest. Contains a "fail" element, which
            is True if a vehicle crashed, and False otherwise

        Returns
        -------
        reward : float or list of float
        """
        return 0





