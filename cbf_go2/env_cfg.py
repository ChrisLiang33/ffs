"""Goal-reaching env for Go2 with frozen flat-locomotion as inner controller.

Outer action = (vx, vy, wz) in body frame. PreTrainedPolicyAction runs the
JIT'd locomotion checkpoint every `low_level_decimation` sub-steps. Four
cylinder obstacles in a plus pattern force the naive u_nom to crash.
"""

import math

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.navigation.mdp as nav_mdp
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg

from . import mdp as cbf_mdp
from .cbf_params_action import CBFParamsActionCfg

LOW_LEVEL_ENV_CFG = UnitreeGo2FlatEnvCfg()
POLICY_CHECKPOINT = (
    "/home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/"
    "2026-05-26_20-00-11/exported/policy.pt"
)

OBSTACLE_RADIUS = 0.5
OBSTACLE_HEIGHT = 1.0
OBSTACLE_POSITIONS = [(1.2, 0.0), (0.0, 1.2), (-1.2, 0.0), (0.0, -1.2)]
OBSTACLE_NAMES = tuple(f"obstacle_{i}" for i in range(len(OBSTACLE_POSITIONS)))
COLLISION_MARGIN = OBSTACLE_RADIUS + 0.15  # obstacle radius + rough robot half-width


@configclass
class CommandsCfg:
    goal_pose = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=True,
        resampling_time_range=(1e9, 1e9),
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-3.0, 3.0), pos_y=(-3.0, 3.0), heading=(-math.pi, math.pi)
        ),
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    cbf_params: CBFParamsActionCfg = CBFParamsActionCfg(
        asset_name="robot",
        inner_cfg=nav_mdp.PreTrainedPolicyActionCfg(
            asset_name="robot",
            policy_path=POLICY_CHECKPOINT,
            low_level_decimation=4,
            low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
            low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
        ),
        obstacle_names=OBSTACLE_NAMES,
        obstacle_radius=OBSTACLE_RADIUS,
        robot_radius=0.3,
        command_name="goal_pose",
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        goal = ObsTerm(func=mdp.generated_commands, params={"command_name": "goal_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    pass


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    goal_reached = DoneTerm(
        func=cbf_mdp.at_goal,
        params={"command_name": "goal_pose", "threshold": 0.3},
    )
    obstacle_hit = DoneTerm(
        func=cbf_mdp.in_collision,
        params={"obstacle_names": OBSTACLE_NAMES, "margin": COLLISION_MARGIN},
    )


@configclass
class GoalGo2EnvCfg(ManagerBasedRLEnvCfg):
    scene = LOW_LEVEL_ENV_CFG.scene
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events = LOW_LEVEL_ENV_CFG.events

    def __post_init__(self):
        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10
        self.episode_length_s = 20.0
        self.scene.num_envs = 4
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        for i, (x, y) in enumerate(OBSTACLE_POSITIONS):
            setattr(
                self.scene,
                f"obstacle_{i}",
                RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Obstacle_{i}",
                    spawn=sim_utils.MeshCylinderCfg(
                        radius=OBSTACLE_RADIUS,
                        height=OBSTACLE_HEIGHT,
                        axis="Z",
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, OBSTACLE_HEIGHT / 2)),
                ),
            )

        # swap the directional arrow for a tall green pole (no orientation ambiguity)
        self.commands.goal_pose.goal_pose_visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Command/pose_goal",
            markers={
                "pole": sim_utils.CylinderCfg(
                    radius=0.08,
                    height=2.0,
                    axis="Z",
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 1.0, 0.1)),
                ),
            },
        )
