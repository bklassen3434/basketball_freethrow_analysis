import numpy as np
import json
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from biomechanics import *
import matplotlib.cm as cm
import matplotlib.colors as colors
from helper import ball_leaves_hands

# Default connections between joints. Change these as you please
connections = [
    # ("R_EYE", "L_EYE"),
    # ("R_EYE", "NOSE"),
    # ("L_EYE", "NOSE"),
    # ("R_EYE", "R_EAR"),
    # ("L_EYE", "L_EAR"),
    ("R_SHOULDER", "L_SHOULDER"),
    ("R_SHOULDER", "R_ELBOW"),
    ("L_SHOULDER", "L_ELBOW"),
    ("R_ELBOW", "R_WRIST"),
    ("L_ELBOW", "L_WRIST"),
    ("R_SHOULDER", "R_HIP"),
    ("L_SHOULDER", "L_HIP"),
    ("R_HIP", "L_HIP"),
    ("R_HIP", "R_KNEE"),
    ("L_HIP", "L_KNEE"),
    ("R_KNEE", "R_ANKLE"),
    ("L_KNEE", "L_ANKLE"),
    ("R_WRIST", "R_1STFINGER"),
    ("R_WRIST", "R_5THFINGER"),
    ("L_WRIST", "L_1STFINGER"),
    ("L_WRIST", "L_5THFINGER"),
    # ("R_ANKLE", "R_1STTOE"),
    # ("R_ANKLE", "R_5THTOE"),
    # ("L_ANKLE", "L_1STTOE"),
    # ("L_ANKLE", "L_5THTOE"),
    ("R_ANKLE", "R_CALC"),
    ("L_ANKLE", "L_CALC"),
    ("R_1STTOE", "R_5THTOE"),
    ("L_1STTOE", "L_5THTOE"),
    ("R_1STTOE", "R_CALC"),
    ("L_1STTOE", "L_CALC"),
    ("R_5THTOE", "R_CALC"),
    ("L_5THTOE", "L_CALC"),
    ("R_1STFINGER", "R_5THFINGER"),
    ("L_1STFINGER", "L_5THFINGER"),
]


def animate_trial(
    trial,
    energy=False,
    efficiency=False,
    alpha=0.5,
    connections=connections,
    xbuffer=4.0,
    ybuffer=4.0,
    zlim=8.0,
    elev=15.0,
    azim=40.0,
    player_color="purple",
    player_lw=4,
    ball_color="#ee6730",
    ball_size=20.0,
    show_court=True,
    notebook_mode=True,
):
    """
    Function to animate a single trial of 3D pose data.

    Parameters:
    -----------
    - trial: int
        Trial number used to build path to json file (0 to 124).
    - energy: bool
        Whether to show the total kinetic energy of each ligament.
    - efficiency: bool
        Whether to show efficiencies at each joint.
    - alpha: float
        The parameter for exponential smoothing
    - connections: list of tuples
        A list of tuples, where each tuple contains two strings representing the joints to connect.
    - xbuffer: float
        The buffer to add to the x-axis limits.
    - ybuffer: float
        The buffer to add to the y-axis limits.
    - zlim: float
        The limit for the z-axis height.
    - elev: float
        The elevation angle for the 3D plot.
    - azim: float
        The azimuth angle for the 3D plot.
    - player_color: str
        The color to use for the player lines.
    - player_lw: float
        The line width to use for the player lines.
    - ball_color: str
        The color to use for the ball.
    - ball_size: float
        The size to use for the ball.
    - show_court: bool
        Whether to show the basketball court in the background.
    - notebook_mode: bool
        Whether function is used within a Jupyter notebook.

    Returns:
    --------
    - anim: matplotlib.animation.FuncAnimation
        The animation object created by the function.
    """

    if notebook_mode:
        plt.rcParams["animation.html"] = "jshtml"

    if show_court:
        try:
            from mplbasketball.court3d import draw_court_3d
        except ModuleNotFoundError:
            print("mplbasketball not installed. Cannot show court.")
            show_court = False

    trial_id = str(trial+1).zfill(4)
    with open(f'./data/P0001/BB_FT_P0001_T{trial_id}.json', "r") as f:
        data = json.load(f)

    player_joint_dict = {}
    ball_data_array = []

    # N_frames = len(data["tracking"])-1

    end_frame = ball_leaves_hands(trial)
    start_frame = end_frame - 30

    joints = ['R_ELBOW', 'L_ELBOW', 'R_KNEE', 'L_KNEE', 'R_HIP', 'L_HIP', 'R_SHOULDER', 'L_SHOULDER']

    # The block of code below returns a dictionary, where each key is a 3D time series for coordinates of a joint.
    # Note that it is a list of N_frames elements, each of which is a 3-element list.
    # If you want to use numpy, you will have to cast it into a numpy array.
    for frame_data in data["tracking"]:
        for joint in frame_data["data"]["player"]:
            if joint not in player_joint_dict:
                player_joint_dict[joint] = []
            player_joint_dict[joint].append(frame_data["data"]["player"][joint])
        ball_data_array.append(frame_data["data"]["ball"])

    # For convenience, we will cast everything to numpy arrays here, but you can keep them as lists if you prefer.
    for joint in player_joint_dict:
        player_joint_dict[joint] = np.array(player_joint_dict[joint])
    ball_data_array = np.array(ball_data_array)

    # Animate the data
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    # Set up initial plot properties
    ax.set_zlim([0, zlim])
    ax.set_box_aspect([1, 1, 1])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=elev, azim=azim)

    

    if energy:
        norm = colors.Normalize(vmin=0, vmax=5)
        cmap = cm.coolwarm
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.04)
        cbar.set_label('Kinetic Energy (J)', rotation=270, labelpad=15)


    if efficiency:
        norm = colors.Normalize(vmin=0, vmax=100)
        cmap = cm.inferno
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.04)
        cbar.set_label('Kinetic Energy Efficiency (%)', rotation=270, labelpad=15)

    # Prepare the lines to be updated
    lines = {
        connection: ax.plot([], [], [], lw=player_lw)[0]
        for connection in connections
    }

    # Prepare the joint efficiencies to be updated
    joint_eff = {
        joint: ax.plot([], [], [], 'o', markersize=12)[0]
        for joint in player_joint_dict
    }

    (ball,) = ax.plot([], [], [], "o", markersize=ball_size, c=ball_color)

    def update(frame):

        # Use the average of the right and left hip to center the view.
        rh_xy = player_joint_dict["R_HIP"][frame][:2]
        lh_xy = player_joint_dict["L_HIP"][frame][:2]
        mh_xy = (rh_xy + lh_xy) / 2

        ax.set_xlim([mh_xy[0] - xbuffer, mh_xy[0] + xbuffer])
        ax.set_ylim([mh_xy[1] - ybuffer, mh_xy[1] + ybuffer])

        # Update the line data for each connection
        for connection in connections:
            part1, part2 = connection
            x = [
                player_joint_dict[part1][frame, 0],
                player_joint_dict[part2][frame, 0],
            ]
            y = [
                player_joint_dict[part1][frame, 1],
                player_joint_dict[part2][frame, 1],
            ]
            z = [
                player_joint_dict[part1][frame, 2],
                player_joint_dict[part2][frame, 2],
            ]
            if energy:
                ke_prev = None
                if ((part1, part2) in sequence) and (frame < end_frame+1) and (frame >= start_frame):
                    body_part = sequence[(part1, part2)]
                    if ke_prev == None:
                        ke_prev = kinetic_energy_total(body_part, part1, part2, frame, trial)
                        ke_curr = ke_prev
                    else:
                        ke_curr = alpha * kinetic_energy_total(body_part, part1, part2, frame, trial) + (1 - alpha) * ke_prev

                    ke_prev = ke_curr

                    color = plt.cm.coolwarm(ke_curr)
                else:
                    color = player_color
            else:
                color = player_color
            lines[connection].set_data_3d(x, y, z)
            lines[connection].set_color(color)

        if efficiency:
            # Plot circles to indicate the efficiency of the energy transfer at each joint
            for joint in player_joint_dict:
                if (joint in joints) and (frame < end_frame+1) and (frame >= start_frame):
                    x, y, z = player_joint_dict[joint][frame, :]

                    ke_eff = kinetic_energy_efficiency_norm(joint, frame, trial)
                    color = plt.cm.inferno(ke_eff)

                    joint_eff[joint].set_data_3d([x], [y], [z])
                    joint_eff[joint].set_color(color)

                elif frame > end_frame+1:
                    joint_eff[joint].set_data_3d([], [], [])

        # Update ball data
        x = ball_data_array[frame, 0]
        y = ball_data_array[frame, 1]
        z = ball_data_array[frame, 2]
        ball.set_data_3d([x], [y], [z])

        # Remove previous frame number if it exists
        if hasattr(ax, 'frame_text'):
            ax.frame_text.remove()

        # Add frame number to the top right corner
        ax.frame_text = ax.text2D(
            0.95, 0.95, f'Frame: {frame}',
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))


        

    if show_court is True:
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("w")
        ax.yaxis.pane.set_edgecolor("w")
        ax.zaxis.pane.set_edgecolor("w")
        ax.xaxis.line.set_linewidth(0)
        ax.yaxis.line.set_linewidth(0)
        ax.zaxis.line.set_linewidth(0)
        draw_court_3d(ax, origin=np.array([0.0, 0.0]), line_width=2)

    # plt.tight_layout()
    plt.subplots(layout="constrained")
    plt.close()

    anim = FuncAnimation(fig, update, frames=range(start_frame, end_frame+1), interval=1000 / 30)
    return anim
