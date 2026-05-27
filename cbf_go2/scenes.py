"""Predefined static obstacle layouts for eval. Body-frame xy (m).

Each scene has up to 4 obstacle positions. Unused slots are pushed off-stage (50, 50).
All scenes assume r=0.5 and no drift (set per-env at eval time).
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


def apply_scene(env, scene_name: str) -> None:
    """Override obstacle positions to the named scene's layout. Static (no drift)."""
    import torch
    layout = SCENES.get(scene_name)
    if layout is None:
        return
    K = 4
    with torch.inference_mode():
        for k, (x, y) in enumerate(layout[:K]):
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
            env._obstacle_velocities.zero_()
        if hasattr(env, "_obstacle_radii"):
            env._obstacle_radii.fill_(0.5)
