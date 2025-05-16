"""Bimanual IK

Same as 01_basic_ik.py, but with two end effectors!
"""

import time
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
import numpy as np
import os
import csv

import pyroki as pk
from viser.extras import ViserUrdf
import pyroki_snippets as pks

from mujoco_ar import MujocoARConnector


def main():
    """Main function for bimanual IK."""

    urdf = load_robot_description("agibot_x1_description")
    target_link_names = ["right_wrist_pitch", "left_wrist_pitch"]

    # Create robot.
    robot = pk.Robot.from_urdf(urdf)

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Initialize previous solution
    previous_solution = np.zeros(robot.joints.num_actuated_joints)

    # Get initial forward kinematics
    initial_fk = robot.forward_kinematics(previous_solution)
    # Get indices of target links
    right_wrist_index = robot.links.names.index("right_wrist_pitch")
    left_wrist_index = robot.links.names.index("left_wrist_pitch")

    # Initialize last known positions for both hands
    right_hand_position = np.array(initial_fk[right_wrist_index][4:7])
    left_hand_position = np.array(initial_fk[left_wrist_index][4:7])

    # Store previous positions for movement capping
    prev_right_hand_position = right_hand_position.copy()
    prev_left_hand_position = left_hand_position.copy()

    # Variables for handling hand toggling
    controller_position_at_right_toggle = np.zeros(3)  # Controller position when last switched from right
    controller_position_at_left_toggle = np.zeros(3)   # Controller position when last switched from left
    right_hand_offset = np.zeros(3)                   # Right hand offset from initial position
    left_hand_offset = np.zeros(3)                    # Left hand offset from initial position
    current_hand_is_right = True                      # Start controlling right hand
    last_toggle_state = False                         # Track toggle changes

    # Maximum allowed movement per frame (in meters)
    max_position_delta = np.array([0.05, 0.05, 0.05])  # Max movement in x, y, z per frame

    # Create interactive controller with initial position.
    ik_target_0 = server.scene.add_transform_controls(
        "/ik_target_0", scale=0.2,
        position=(initial_fk[right_wrist_index][4], initial_fk[right_wrist_index][5], initial_fk[right_wrist_index][6]),
        wxyz=(initial_fk[right_wrist_index][0], initial_fk[right_wrist_index][1], initial_fk[right_wrist_index][2], initial_fk[right_wrist_index][3])
    )
    ik_target_1 = server.scene.add_transform_controls(
        "/ik_target_1", scale=0.2,
        position=(initial_fk[left_wrist_index][4], initial_fk[left_wrist_index][5], initial_fk[left_wrist_index][6]),
        wxyz=(initial_fk[left_wrist_index][0], initial_fk[left_wrist_index][1], initial_fk[left_wrist_index][2], initial_fk[left_wrist_index][3])
    )
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
    
    # Add a slider to control the smoothness weight
    smoothness_slider = server.gui.add_slider(
        "Smoothness Weight", 
        min=0.0, 
        max=100.0,
        step=1.0, 
        initial_value=50.0,
    )

    connector = MujocoARConnector()
    connector.start()

    while True:
        data = connector.get_latest_data()
        if data["position"] is None:
            continue

        # Calculate target position based on controller movement
        controller_position = np.asarray(data["position"])
        scaling = np.array([2, 2, 2])

        # Check if toggle state changed
        current_toggle = data.get("toggle", False)
        if current_toggle != last_toggle_state:
            if current_toggle:  # Switching from right to left
                # Save controller position and current right hand offset
                controller_position_at_right_toggle = controller_position
                right_hand_offset = right_hand_position - initial_fk[right_wrist_index][4:7]
                current_hand_is_right = False
            else:  # Switching from left to right
                # Save controller position and current left hand offset
                controller_position_at_left_toggle = controller_position
                left_hand_offset = left_hand_position - initial_fk[left_wrist_index][4:7]
                current_hand_is_right = True

            last_toggle_state = current_toggle

        # Calculate hand positions based on controller movement and saved offsets
        if current_hand_is_right:
            # For right hand: apply controller movement since last toggle from left
            controller_movement = controller_position - controller_position_at_left_toggle
            # Update right hand position (preserve the offset it had when we toggled away)
            new_right_hand_pos = initial_fk[right_wrist_index][4:7] + right_hand_offset + scaling * controller_movement
            right_hand_position = np.array(new_right_hand_pos)
            active_position = right_hand_position
            inactive_position = left_hand_position
        else:
            # For left hand: apply controller movement since last toggle from right
            controller_movement = controller_position - controller_position_at_right_toggle
            # Update left hand position (preserve the offset it had when we toggled away)
            new_left_hand_pos = initial_fk[left_wrist_index][4:7] + left_hand_offset + scaling * controller_movement
            left_hand_position = np.array(new_left_hand_pos)
            active_position = left_hand_position
            inactive_position = right_hand_position

        # Cap positions to prevent extreme movements between frames
        # Right hand capping
        delta_right = right_hand_position - prev_right_hand_position
        capped_right_hand = right_hand_position.copy()
        for i in range(3):
            if abs(delta_right[i]) > max_position_delta[i]:
                capped_right_hand[i] = prev_right_hand_position[i] + np.sign(delta_right[i]) * max_position_delta[i]
        right_hand_position = capped_right_hand

        # Left hand capping
        delta_left = left_hand_position - prev_left_hand_position
        capped_left_hand = left_hand_position.copy()
        for i in range(3):
            if abs(delta_left[i]) > max_position_delta[i]:
                capped_left_hand[i] = prev_left_hand_position[i] + np.sign(delta_left[i]) * max_position_delta[i]
        left_hand_position = capped_left_hand

        # Update active/inactive positions after capping
        if current_hand_is_right:
            active_position = right_hand_position
            inactive_position = left_hand_position
        else:
            active_position = left_hand_position
            inactive_position = right_hand_position

        # Debug output
        print(f"Active hand: {'Right' if current_hand_is_right else 'Left'}")
        print(f"Controller: {controller_position}, Movement: {controller_movement}")

        # Solve IK.
        start_time = time.time()
        
        # Use the right positions based on which hand is active
        if not data.get("toggle", False):
            target_positions = np.array([active_position, inactive_position])
        else:
            target_positions = np.array([inactive_position, active_position])

        solution = pks.solve_ik_with_multiple_targets(
            robot=robot,
            target_link_names=target_link_names,
            target_positions=target_positions,
            target_wxyzs=np.array([ik_target_0.wxyz, ik_target_1.wxyz]),
            prior_configuration=previous_solution,
            smoothness_weight=smoothness_slider.value,
        )
        print("solution", solution)
        
        # Save current solution for next iteration
        previous_solution = solution

        # Update timing handle.
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        # Update visualizer.
        urdf_vis.update_cfg(solution)

        # Write solution to CSV file (overwrite previous content)
        with open(os.path.expanduser("~/pose.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(solution)

        # Store current positions as previous for next frame
        prev_right_hand_position = right_hand_position.copy()
        prev_left_hand_position = left_hand_position.copy()


if __name__ == "__main__":
    main()
