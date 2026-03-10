import time
from typing import Sequence

import numpy as np
import pybullet as p


DEFAULT_CONTROL_DT = 1.0 / 120.0


def _step_sim(control_dt: float, steps: int = 1) -> None:
    for _ in range(max(1, int(steps))):
        p.stepSimulation()
        time.sleep(control_dt)


def _move_linear_quat(
    panda,
    start_pos: Sequence[float],
    end_pos: Sequence[float],
    ee_quaternion: Sequence[float],
    steps: int,
    control_dt: float,
    position_gain: float = 0.04,
) -> None:
    start = np.array(start_pos, dtype=float)
    end = np.array(end_pos, dtype=float)
    for alpha in np.linspace(0.0, 1.0, max(2, int(steps))):
        waypoint = (1.0 - alpha) * start + alpha * end
        panda.move_to_pose(waypoint.tolist(), ee_quaternion=list(ee_quaternion), positionGain=position_gain)
        _step_sim(control_dt, 1)


def go_home(panda, control_dt: float = DEFAULT_CONTROL_DT):
    """Move the end-effector to a neutral pose and open the gripper."""
    home_pose = np.array([0.35, 0.0, 0.45], dtype=float)
    home_yaw = 0.0

    panda.open_gripper()
    _step_sim(control_dt, 30)

    for _ in range(540):
        panda.move_to_pose(home_pose.tolist(), ee_rotz=home_yaw, positionGain=0.04)
        _step_sim(control_dt, 1)

    return panda.get_state()


def open_microwave(
    panda,
    microwave,
    control_dt: float = DEFAULT_CONTROL_DT,
    descend_steps: int = 120,
    move_steps: int = 160,
    rotate_steps: int = 120,
    pull_steps: int = 840,
    standoff_distance: float = 0.14,
    pre_grasp_distance: float = 0.07,
    grasp_distance: float = 0.005,
    grasp_side_offset: float = 0.0035,
    pull_distance_1: float = 0.04,
    pull_distance_2: float = 0.15,
    late_clearance_max: float = 0.01,
    final_follow_steps: int = 240,
):
    """Approach, grasp, and apply a two-stage microwave pull."""
    robot_state = panda.get_state()
    microwave_state = microwave.get_state()

    ee_pos = np.array(robot_state["ee-position"], dtype=float)
    ee_quat = list(robot_state["ee-quaternion"])
    handle_pos = np.array(microwave_state["handle_position"], dtype=float)
    handle_z = float(handle_pos[2])
    base_quat = microwave_state["base_quaternion"]
    base_rot = np.array(p.getMatrixFromQuaternion(base_quat)).reshape(3, 3)

    approach_dir = -np.array(base_rot[:, 0], dtype=float)
    approach_dir[2] = 0.0
    approach_norm = float(np.linalg.norm(approach_dir))
    if approach_norm < 1e-6:
        approach_dir = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        approach_dir = approach_dir / approach_norm

    pull_dir = np.array(base_rot[:, 0] + 0.35 * base_rot[:, 1], dtype=float)
    pull_dir[2] = 0.0
    pull_norm = float(np.linalg.norm(pull_dir))
    if pull_norm < 1e-6:
        pull_dir = approach_dir.copy()
    else:
        pull_dir = pull_dir / pull_norm

    side_dir = np.array(base_rot[:, 1], dtype=float)
    side_dir[2] = 0.0
    side_norm = float(np.linalg.norm(side_dir))
    if side_norm < 1e-6:
        side_dir = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        side_dir = side_dir / side_norm

    descend_pos = np.array([ee_pos[0], ee_pos[1], handle_z], dtype=float)
    standoff_pos = np.array(
        [
            handle_pos[0] - approach_dir[0] * standoff_distance,
            handle_pos[1] - approach_dir[1] * standoff_distance,
            handle_z,
        ],
        dtype=float,
    )

    grasp_yaw = float(np.arctan2(approach_dir[0], -approach_dir[1]))
    grasp_quat = p.getQuaternionFromEuler([np.pi / 2.0, np.pi / 2.0, grasp_yaw])
    start_angle = float(microwave_state["joint_angle"])
    joint_info = p.getJointInfo(microwave.object, 0)
    joint_lower = float(joint_info[8])
    joint_upper = float(joint_info[9])
    open_range = max(abs(joint_lower - start_angle), abs(joint_upper - start_angle), 1e-6)

    panda.open_gripper()
    _step_sim(control_dt, 40)

    _move_linear_quat(
        panda,
        ee_pos,
        descend_pos,
        ee_quaternion=ee_quat,
        steps=descend_steps,
        control_dt=control_dt,
        position_gain=0.05,
    )
    _move_linear_quat(
        panda,
        descend_pos,
        standoff_pos,
        ee_quaternion=ee_quat,
        steps=move_steps,
        control_dt=control_dt,
        position_gain=0.05,
    )

    for _ in range(max(2, int(rotate_steps))):
        panda.move_to_pose(standoff_pos.tolist(), ee_quaternion=list(grasp_quat), positionGain=0.08)
        _step_sim(control_dt, 1)

    microwave_state = microwave.get_state()
    handle_pos = np.array(microwave_state["handle_position"], dtype=float)
    pre_grasp_pos = np.array(
        [
            handle_pos[0] - approach_dir[0] * pre_grasp_distance + side_dir[0] * grasp_side_offset,
            handle_pos[1] - approach_dir[1] * pre_grasp_distance + side_dir[1] * grasp_side_offset,
            handle_z,
        ],
        dtype=float,
    )
    grasp_pos = np.array(
        [
            handle_pos[0] - approach_dir[0] * grasp_distance + side_dir[0] * grasp_side_offset,
            handle_pos[1] - approach_dir[1] * grasp_distance + side_dir[1] * grasp_side_offset,
            handle_z,
        ],
        dtype=float,
    )

    _move_linear_quat(
        panda,
        standoff_pos,
        pre_grasp_pos,
        ee_quaternion=grasp_quat,
        steps=max(2, move_steps // 2),
        control_dt=control_dt,
        position_gain=0.06,
    )
    _move_linear_quat(
        panda,
        pre_grasp_pos,
        grasp_pos,
        ee_quaternion=grasp_quat,
        steps=max(2, move_steps // 2),
        control_dt=control_dt,
        position_gain=0.06,
    )

    for _ in range(180):
        panda.move_to_pose(grasp_pos.tolist(), ee_quaternion=list(grasp_quat), positionGain=0.08)
        panda.close_gripper()
        _step_sim(control_dt, 1)

    handle_state = microwave.get_state()
    handle_now = list(handle_state["handle_position"])
    handle_quat = list(handle_state["handle_quaternion"])
    ee_state = panda.get_state()
    ee_now = list(ee_state["ee-position"])
    ee_quat_now = list(ee_state["ee-quaternion"])
    inv_handle_pos, inv_handle_quat = p.invertTransform(handle_now, handle_quat)
    handle_to_ee_pos, handle_to_ee_quat = p.multiplyTransforms(
        inv_handle_pos,
        inv_handle_quat,
        ee_now,
        ee_quat_now,
    )
    lead_offset = np.zeros(3, dtype=float)

    stage1_steps = max(2, int(pull_steps // 2))
    stage1_delta = pull_dir * (pull_distance_1 / stage1_steps)
    for _ in range(stage1_steps):
        handle_state = microwave.get_state()
        current_angle = float(handle_state["joint_angle"])
        if current_angle <= joint_lower + 0.03:
            break
        progress = min(1.0, abs(current_angle - start_angle) / open_range)
        if progress <= 0.55:
            slow_scale = 1.0
        else:
            slow_scale = max(0.25, 1.0 - (progress - 0.55) / 0.45)
        lead_offset = lead_offset + stage1_delta * slow_scale
        handle_now = np.array(handle_state["handle_position"], dtype=float)
        handle_quat = list(handle_state["handle_quaternion"])
        clearance = max(0.0, (progress - 0.55) / 0.45) * late_clearance_max
        adjusted_local = (np.array(handle_to_ee_pos, dtype=float) + np.array([clearance, 0.0, 0.0], dtype=float)).tolist()
        desired_handle_pos = (handle_now + lead_offset).tolist()
        follow_pos, follow_quat = p.multiplyTransforms(
            desired_handle_pos,
            handle_quat,
            adjusted_local,
            handle_to_ee_quat,
        )
        pull_gain = 0.02 + 0.04 * slow_scale
        panda.move_to_pose(list(follow_pos), ee_quaternion=list(follow_quat), positionGain=pull_gain)
        panda.close_gripper()
        _step_sim(control_dt, 1)

    stage2_steps = max(2, int(pull_steps // 2))
    stage2_delta = -side_dir * (pull_distance_2 / stage2_steps)
    for _ in range(stage2_steps):
        handle_state = microwave.get_state()
        current_angle = float(handle_state["joint_angle"])
        if current_angle <= joint_lower + 0.03:
            break
        progress = min(1.0, abs(current_angle - start_angle) / open_range)
        if progress <= 0.45:
            slow_scale = 1.0
        else:
            slow_scale = max(0.1, 1.0 - (progress - 0.45) / 0.55)
        lead_offset = lead_offset + stage2_delta * slow_scale
        handle_now = np.array(handle_state["handle_position"], dtype=float)
        handle_quat = list(handle_state["handle_quaternion"])
        clearance = max(0.0, (progress - 0.45) / 0.55) * late_clearance_max
        adjusted_local = (np.array(handle_to_ee_pos, dtype=float) + np.array([clearance, 0.0, 0.0], dtype=float)).tolist()
        desired_handle_pos = (handle_now + lead_offset).tolist()
        follow_pos, follow_quat = p.multiplyTransforms(
            desired_handle_pos,
            handle_quat,
            adjusted_local,
            handle_to_ee_quat,
        )
        pull_gain = 0.012 + 0.028 * slow_scale
        panda.move_to_pose(list(follow_pos), ee_quaternion=list(follow_quat), positionGain=pull_gain)
        panda.close_gripper()
        _step_sim(control_dt, 1)

    for _ in range(max(2, int(final_follow_steps))):
        handle_state = microwave.get_state()
        current_angle = float(handle_state["joint_angle"])
        if current_angle <= joint_lower + 0.03:
            break
        progress = min(1.0, abs(current_angle - start_angle) / open_range)
        handle_now = np.array(handle_state["handle_position"], dtype=float)
        handle_quat = list(handle_state["handle_quaternion"])
        clearance = max(0.0, (progress - 0.4) / 0.6) * late_clearance_max
        adjusted_local = (np.array(handle_to_ee_pos, dtype=float) + np.array([clearance, 0.0, 0.0], dtype=float)).tolist()
        desired_handle_pos = (handle_now + lead_offset).tolist()
        follow_pos, follow_quat = p.multiplyTransforms(
            desired_handle_pos,
            handle_quat,
            adjusted_local,
            handle_to_ee_quat,
        )
        follow_gain = 0.01
        panda.move_to_pose(list(follow_pos), ee_quaternion=list(follow_quat), positionGain=follow_gain)
        panda.close_gripper()
        _step_sim(control_dt, 1)

    final_angle = float(microwave.get_state()["joint_angle"])
    return {"start_angle_rad": start_angle, "final_angle_rad": final_angle}


def pick_cube(
    panda,
    cube,
    control_dt: float = DEFAULT_CONTROL_DT,
    approach_height: float = 0.12,
    grasp_height: float = 0.0,
):
    """Pick a cube from the table using a top-down grasp."""
    cube_state = cube.get_state()
    cube_pos = np.array(cube_state["position"], dtype=float)
    cube_yaw = cube_state["euler"][2]
    ee_yaw = cube_yaw

    approach_pos = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + approach_height], dtype=float)
    grasp_pos = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + grasp_height], dtype=float)
    lift_pos = approach_pos + np.array([0.0, 0.0, 0.08], dtype=float)

    panda.open_gripper()
    _step_sim(control_dt, 40)

    for _ in range(360):
        panda.move_to_pose(approach_pos.tolist(), ee_rotz=ee_yaw, positionGain=0.05)
        _step_sim(control_dt, 1)

    for _ in range(360):
        panda.move_to_pose(grasp_pos.tolist(), ee_rotz=ee_yaw, positionGain=0.05)
        _step_sim(control_dt, 1)

    for _ in range(180):
        panda.move_to_pose(grasp_pos.tolist(), ee_rotz=ee_yaw, positionGain=0.05)
        _step_sim(control_dt, 1)

    panda.close_gripper()
    _step_sim(control_dt, 60)

    for _ in range(360):
        panda.move_to_pose(lift_pos.tolist(), ee_rotz=ee_yaw, positionGain=0.05)
        _step_sim(control_dt, 1)

    return cube.get_state()["position"]


def open_cabinet(
    panda,
    cabinet,
    control_dt: float = DEFAULT_CONTROL_DT,
    descend_steps: int = 120,
    move_steps: int = 160,
    rotate_steps: int = 120,
    standoff_distance: float = 0.16,
    pre_grasp_distance: float = 0.08,
    grasp_distance: float = 0.005,
    grasp_hold_steps: int = 180,
    pull_steps: int = 480,
    target_cabinet_extension: float = 0.14,
):
    """Approach the cabinet handle, grasp it, and pull the drawer open."""
    robot_state = panda.get_state()
    cabinet_state = cabinet.get_state()

    ee_pos = np.array(robot_state["ee-position"], dtype=float)
    ee_quat = list(robot_state["ee-quaternion"])
    handle_pos = np.array(cabinet_state["handle_position"], dtype=float)
    handle_z = float(handle_pos[2])
    base_quat = cabinet_state["base_quaternion"]
    base_rot = np.array(p.getMatrixFromQuaternion(base_quat)).reshape(3, 3)

    slide_dir = np.array(base_rot[:, 0], dtype=float)
    slide_dir[2] = 0.0
    slide_norm = float(np.linalg.norm(slide_dir))
    if slide_norm < 1e-6:
        slide_dir = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        slide_dir = slide_dir / slide_norm
    approach_dir = -slide_dir

    descend_pos = np.array([ee_pos[0], ee_pos[1], handle_z], dtype=float)
    standoff_pos = np.array(
        [
            handle_pos[0] - approach_dir[0] * standoff_distance,
            handle_pos[1] - approach_dir[1] * standoff_distance,
            handle_z,
        ],
        dtype=float,
    )

    grasp_yaw = float(np.arctan2(approach_dir[0], -approach_dir[1]))
    grasp_quat = p.getQuaternionFromEuler([np.pi / 2.0, 0.0, grasp_yaw])
    start_extension = float(cabinet_state["joint_angle"])

    panda.open_gripper()
    _step_sim(control_dt, 40)

    _move_linear_quat(
        panda,
        ee_pos,
        descend_pos,
        ee_quaternion=ee_quat,
        steps=descend_steps,
        control_dt=control_dt,
        position_gain=0.05,
    )
    _move_linear_quat(
        panda,
        descend_pos,
        standoff_pos,
        ee_quaternion=ee_quat,
        steps=move_steps,
        control_dt=control_dt,
        position_gain=0.05,
    )

    for _ in range(max(2, int(rotate_steps))):
        panda.move_to_pose(standoff_pos.tolist(), ee_quaternion=list(grasp_quat), positionGain=0.08)
        _step_sim(control_dt, 1)

    handle_pos = np.array(cabinet.get_state()["handle_position"], dtype=float)
    pre_grasp_pos = np.array(
        [
            handle_pos[0] - approach_dir[0] * pre_grasp_distance,
            handle_pos[1] - approach_dir[1] * pre_grasp_distance,
            handle_z,
        ],
        dtype=float,
    )
    grasp_pos = np.array(
        [
            handle_pos[0] - approach_dir[0] * grasp_distance,
            handle_pos[1] - approach_dir[1] * grasp_distance,
            handle_z,
        ],
        dtype=float,
    )

    _move_linear_quat(
        panda,
        standoff_pos,
        pre_grasp_pos,
        ee_quaternion=grasp_quat,
        steps=max(2, move_steps // 2),
        control_dt=control_dt,
        position_gain=0.06,
    )
    _move_linear_quat(
        panda,
        pre_grasp_pos,
        grasp_pos,
        ee_quaternion=grasp_quat,
        steps=max(2, move_steps // 2),
        control_dt=control_dt,
        position_gain=0.06,
    )

    for _ in range(max(2, int(grasp_hold_steps))):
        panda.move_to_pose(grasp_pos.tolist(), ee_quaternion=list(grasp_quat), positionGain=0.08)
        panda.close_gripper()
        _step_sim(control_dt, 1)

    joint_info = p.getJointInfo(cabinet.object, 0)
    lower_limit = float(joint_info[8])
    upper_limit = float(joint_info[9])
    start_extension = float(np.clip(cabinet.get_state()["joint_angle"], lower_limit, upper_limit))
    clamped_target = float(np.clip(target_cabinet_extension, lower_limit, upper_limit))
    pull_distance = max(0.0, clamped_target - start_extension)

    pull_start = np.array(panda.get_state()["ee-position"], dtype=float)
    pull_pos = pull_start + slide_dir * pull_distance
    for alpha in np.linspace(0.0, 1.0, max(2, int(pull_steps))):
        waypoint = (1.0 - alpha) * pull_start + alpha * pull_pos
        panda.move_to_pose(waypoint.tolist(), ee_quaternion=list(grasp_quat), positionGain=0.05)
        panda.close_gripper()
        _step_sim(control_dt, 1)

    final_extension = float(cabinet.get_state()["joint_angle"])
    return {"start_extension_m": start_extension, "target_extension_m": clamped_target, "final_extension_m": final_extension}


def close_cabinet(
    panda,
    cabinet,
    control_dt: float = DEFAULT_CONTROL_DT,
    descend_steps: int = 120,
    move_steps: int = 160,
    rotate_steps: int = 120,
    standoff_distance: float = 0.16,
    pre_grasp_distance: float = 0.08,
    grasp_distance: float = 0.005,
    grasp_hold_steps: int = 180,
    push_steps: int = 480,
    target_cabinet_extension: float = 0.0,
):
    """Approach the cabinet handle, grasp it, and push the drawer closed."""
    robot_state = panda.get_state()
    cabinet_state = cabinet.get_state()

    ee_pos = np.array(robot_state["ee-position"], dtype=float)
    ee_quat = list(robot_state["ee-quaternion"])
    handle_pos = np.array(cabinet_state["handle_position"], dtype=float)
    handle_z = float(handle_pos[2])
    base_quat = cabinet_state["base_quaternion"]
    base_rot = np.array(p.getMatrixFromQuaternion(base_quat)).reshape(3, 3)

    slide_dir = np.array(base_rot[:, 0], dtype=float)
    slide_dir[2] = 0.0
    slide_norm = float(np.linalg.norm(slide_dir))
    if slide_norm < 1e-6:
        slide_dir = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        slide_dir = slide_dir / slide_norm
    approach_dir = -slide_dir

    descend_pos = np.array([ee_pos[0], ee_pos[1], handle_z], dtype=float)
    standoff_pos = np.array(
        [
            handle_pos[0] - approach_dir[0] * standoff_distance,
            handle_pos[1] - approach_dir[1] * standoff_distance,
            handle_z,
        ],
        dtype=float,
    )

    grasp_yaw = float(np.arctan2(approach_dir[0], -approach_dir[1]))
    grasp_quat = p.getQuaternionFromEuler([np.pi / 2.0, 0.0, grasp_yaw])
    start_extension = float(cabinet_state["joint_angle"])

    panda.open_gripper()
    _step_sim(control_dt, 40)

    _move_linear_quat(
        panda,
        ee_pos,
        descend_pos,
        ee_quaternion=ee_quat,
        steps=descend_steps,
        control_dt=control_dt,
        position_gain=0.05,
    )
    _move_linear_quat(
        panda,
        descend_pos,
        standoff_pos,
        ee_quaternion=ee_quat,
        steps=move_steps,
        control_dt=control_dt,
        position_gain=0.05,
    )

    for _ in range(max(2, int(rotate_steps))):
        panda.move_to_pose(standoff_pos.tolist(), ee_quaternion=list(grasp_quat), positionGain=0.08)
        _step_sim(control_dt, 1)

    handle_pos = np.array(cabinet.get_state()["handle_position"], dtype=float)
    pre_grasp_pos = np.array(
        [
            handle_pos[0] - approach_dir[0] * pre_grasp_distance,
            handle_pos[1] - approach_dir[1] * pre_grasp_distance,
            handle_z,
        ],
        dtype=float,
    )
    grasp_pos = np.array(
        [
            handle_pos[0] - approach_dir[0] * grasp_distance,
            handle_pos[1] - approach_dir[1] * grasp_distance,
            handle_z,
        ],
        dtype=float,
    )

    _move_linear_quat(
        panda,
        standoff_pos,
        pre_grasp_pos,
        ee_quaternion=grasp_quat,
        steps=max(2, move_steps // 2),
        control_dt=control_dt,
        position_gain=0.06,
    )
    _move_linear_quat(
        panda,
        pre_grasp_pos,
        grasp_pos,
        ee_quaternion=grasp_quat,
        steps=max(2, move_steps // 2),
        control_dt=control_dt,
        position_gain=0.06,
    )

    for _ in range(max(2, int(grasp_hold_steps))):
        panda.move_to_pose(grasp_pos.tolist(), ee_quaternion=list(grasp_quat), positionGain=0.08)
        panda.close_gripper()
        _step_sim(control_dt, 1)

    joint_info = p.getJointInfo(cabinet.object, 0)
    lower_limit = float(joint_info[8])
    upper_limit = float(joint_info[9])
    start_extension = float(np.clip(cabinet.get_state()["joint_angle"], lower_limit, upper_limit))
    clamped_target = float(np.clip(target_cabinet_extension, lower_limit, upper_limit))
    push_distance = max(0.0, start_extension - clamped_target)

    push_start = np.array(panda.get_state()["ee-position"], dtype=float)
    push_pos = push_start - slide_dir * push_distance
    for alpha in np.linspace(0.0, 1.0, max(2, int(push_steps))):
        waypoint = (1.0 - alpha) * push_start + alpha * push_pos
        panda.move_to_pose(waypoint.tolist(), ee_quaternion=list(grasp_quat), positionGain=0.05)
        panda.close_gripper()
        _step_sim(control_dt, 1)

    final_extension = float(cabinet.get_state()["joint_angle"])
    return {"start_extension_m": start_extension, "target_extension_m": clamped_target, "final_extension_m": final_extension}
