"""Predefined obstacle layouts for eval. Body-frame xy (m).

Static scenes have just (x, y) positions and zero drift.
Moving scenes have (x, y, vx, vy) so we can re-apply positions on reset and
keep the velocities for `advance_obstacles` to integrate every sub-step.
"""

OFF = (50.0, 50.0)

SCENES: dict[str, list[tuple[float, float]] | None] = {
    "in_dist":  None,  # use the env's random layout (not overridden)
    "open":     [OFF, OFF, OFF, OFF],
    "sparse":   [(2.0, 0.0), (-2.0, 0.0), OFF, OFF],
    "corridor": [(1.5, 1.0), (1.5, -1.0), (-1.5, 1.0), (-1.5, -1.0)],
    "slalom":   [(1.2, 0.7), (2.0, -0.7), (2.8, 0.7), (-1.5, 0.0)],
    "narrow":   [(2.0, 0.4), (2.0, -0.4), OFF, OFF],
    "gauntlet": [(1.5, 0.0), (2.0, 0.6), (2.0, -0.6), (2.5, 0.0)],
}

# Moving scenes: (x, y, vx, vy) per obstacle. Velocities in m/s (world frame).
# Adaptive policies should win here because fixed margins can't anticipate
# fast-approaching obstacles.
MOVING_SCENES: dict[str, list[tuple[float, float, float, float]]] = {
    # Two obstacles drifting toward the centerline at ~1 m/s — classic crossing problem.
    "crossing": [
        (1.5, -1.8, 0.0,  1.0),   # ahead-right, moving up at 1 m/s
        (1.5,  1.8, 0.0, -1.0),   # ahead-left,  moving down at 1 m/s
        (OFF[0], OFF[1], 0.0, 0.0),
        (OFF[0], OFF[1], 0.0, 0.0),
    ],
    # Single obstacle crossing the robot's path laterally at 1.2 m/s.
    # Robot is at (0,0); obstacle starts off to the right, sweeps left.
    "traversing": [
        (2.0,  2.5, 0.0, -1.2),
        (OFF[0], OFF[1], 0.0, 0.0),
        (OFF[0], OFF[1], 0.0, 0.0),
        (OFF[0], OFF[1], 0.0, 0.0),
    ],
    # Single obstacle approaching the robot head-on at 0.8 m/s.
    # Tests anticipating closing-velocity.
    "head_on": [
        (3.0, 0.0, -0.8, 0.0),
        (OFF[0], OFF[1], 0.0, 0.0),
        (OFF[0], OFF[1], 0.0, 0.0),
        (OFF[0], OFF[1], 0.0, 0.0),
    ],
    # Two obstacles closing on each other (gap shrinks over time).
    # Robot must thread the closing gap — pure coordination test.
    "squeeze": [
        (1.5,  1.5, 0.0, -0.8),
        (1.5, -1.5, 0.0,  0.8),
        (OFF[0], OFF[1], 0.0, 0.0),
        (OFF[0], OFF[1], 0.0, 0.0),
    ],
    # Three obstacles in alternating-direction zigzag motion.
    # Tests adapting to multi-trajectory pattern.
    "weaving": [
        (1.0,  0.8, 0.0, -0.6),
        (2.0, -0.8, 0.0,  0.6),
        (3.0,  0.8, 0.0, -0.6),
        (OFF[0], OFF[1], 0.0, 0.0),
    ],
    # Static gauntlet layout but all four obstacles drifting slowly in random directions.
    # Tests dense + dynamic combined.
    "gauntlet_moving": [
        (1.5,  0.0,  0.0,  0.5),
        (2.0,  0.6,  0.4, -0.3),
        (2.0, -0.6, -0.4,  0.3),
        (2.5,  0.0,  0.0, -0.5),
    ],
}


def apply_scene(env, scene_name: str, force_init_positions: bool = False) -> None:
    """Override obstacle positions (+ velocities for moving scenes).

    Static scenes re-apply positions every step (idempotent, since obstacles don't move).
    Moving scenes only re-apply positions when force_init_positions=True (first eval step or
    after a known reset) so `advance_obstacles` integration isn't clobbered each step.
    """
    import torch
    static_layout = SCENES.get(scene_name)
    moving_layout = MOVING_SCENES.get(scene_name)
    if static_layout is None and moving_layout is None:
        return

    K = 4
    with torch.inference_mode():
        if moving_layout is not None:
            for k, (x, y, vx, vy) in enumerate(moving_layout[:K]):
                name = f"obstacle_{k}"
                try:
                    obstacle = env.scene[name]
                except KeyError:
                    continue
                if force_init_positions:
                    new_pose = obstacle.data.default_root_state[:, :7].clone()
                    new_pose[:, 0] = x + env.scene.env_origins[:, 0]
                    new_pose[:, 1] = y + env.scene.env_origins[:, 1]
                    obstacle.write_root_pose_to_sim(new_pose)
                if hasattr(env, "_obstacle_velocities"):
                    env._obstacle_velocities[:, k, 0] = vx
                    env._obstacle_velocities[:, k, 1] = vy
        else:
            for k, (x, y) in enumerate(static_layout[:K]):
                name = f"obstacle_{k}"
                try:
                    obstacle = env.scene[name]
                except KeyError:
                    continue
                new_pose = obstacle.data.default_root_state[:, :7].clone()
                new_pose[:, 0] = x + env.scene.env_origins[:, 0]
                new_pose[:, 1] = y + env.scene.env_origins[:, 1]
                obstacle.write_root_pose_to_sim(new_pose)
                if hasattr(env, "_obstacle_velocities"):
                    env._obstacle_velocities[:, k].zero_()
        if hasattr(env, "_obstacle_radii"):
            env._obstacle_radii.fill_(0.5)
