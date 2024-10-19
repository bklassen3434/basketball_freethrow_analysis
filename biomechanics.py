
##### Biomechanical Functions #####

# Import Packages
import math
import json
import numpy as np
from helper import *



# Loading Data (get data from every free throw)
data = []

# Loop through trial numbers 1 to 125 and load corresponding JSON files
for trial_number in range(1,126):
    trial_id = str(trial_number).zfill(4) # Format trial number with leading zeros

    # Open and load JSON data for the current trial
    with open(f'./data/P0001/BB_FT_P0001_T{trial_id}.json') as json_file:
        free_throw_data = json.load(json_file)
        data.append(free_throw_data)

# Grab only data from around the actual shot (30 second before ball release to 15 seconds after)
data = align_tracking_data(data)



def position(joint, frame, trial):
    '''
    Get the position of a specific body joint at a given frame and trial.

    Parameters:
    joint (str): The name of the joint for which to retrieve the position.
    frame (int): The frame number within the trial to evaluate.
    trial (int): The trial number from which to obtain the joint's position.

    Returns:
    list: A list containing the x, y, and z coordinates of the joint position in metres.
    '''

    return [p / 3.281 for p in data[trial]['tracking'][frame]['data']['player'][joint]] # Convert feet to metres



def time(frame, trial):
    '''
    Get the time associated with a specific frame in a given trial.

    Parameters:
    frame (int): The frame number to retrieve the time for.
    trial (int): The trial number from which to obtain the frame's time.

    Returns:
    float: The time corresponding to the specified frame in seconds.
    '''

    return data[trial]['tracking'][frame]['time'] / 1000 # Convert milliseconds to seconds



def distance(joint1, joint2, frame, trial):
    '''
    Calculate the distance between two specified joints at a given frame and trial.

    Parameters:
    joint1 (str): The name of the first joint.
    joint2 (str): The name of the second joint.
    frame (int): The frame number in which to calculate the distance.
    trial (int): The trial number being evaluated.

    Returns:
    list: A list containing the x, y, and z distances between the two joints.
    '''

    return [p1 - p2 for p1, p2 in zip(position(joint1, frame, trial), position(joint2, frame, trial))]



def velocity_trans(joint, frame, trial):
    '''
    Estimate the translational velocity of a specific joint at a given frame in a trial.

    Parameters:
    joint (str): The name of the joint for which to calculate velocity.
    frame (int): The frame number to evaluate.
    trial (int): The trial number being analyzed.

    Returns:
    list: A list containing the x, y, and z components of the joint's velocity.
    '''

    pos_start = position(joint, frame, trial) # Position at the current frame
    pos_end = position(joint, frame + 1, trial) # Position at the next frame

    t_start = time(frame, trial) # Time at the current frame
    t_end = time(frame + 1, trial) # Time at the next frame

    return [((pos_end[i] - pos_start[i]) / (t_end - t_start)) for i in range(3)] # Calculate velocity



def mass(body_part):
    '''
    Estimate the mass of a specific body part based on a predefined total mass.

    Parameters:
    body_part (str): The name of the body part for which to estimate mass.

    Returns:
    float: The estimated mass of the specified body part in kilograms.
    '''

    # Get the total mass (in kg)
    total_mass = 90.7
    # Estimate the percentage of total mass for each body part
    prop_mass = {'upper_arm': 0.0271,
                 'forearm': 0.0162,

                 'hand_thumbside': 0.00244,
                 'hand_palm': 0.00244,
                 'hand_pinkyside': 0.001525,

                 'thigh': 0.105,
                 'lower_leg': 0.0465,

                 'hindfoot': 0.0058,
                 'midfoot': 0.0029,
                 'forefoot': 0.0029,

                 'head': 0.0694,
                 'trunk_vertical': 0.13038,
                 'trunk_horizontal': 0.08692,}
    
    return total_mass * prop_mass[body_part] # Return mass for the specific body part



def kinetic_energy_trans(body_part, joint1, joint2, frame, trial):
    '''
    Estimate the translational kinetic energy of a body part defined by two joints.

    Parameters:
    body_part (str): The name of the body part for which to calculate kinetic energy.
    joint1 (str): The name of the first joint that defines the body part.
    joint2 (str): The name of the second joint that defines the body part.
    frame (int): The frame number to evaluate.
    trial (int): The trial number being analyzed.

    Returns:
    float: The estimated translational kinetic energy of the specified body part in joules.
    '''

    # Calculate the velocity of each connecting joint
    vel1 = velocity_trans(joint1, frame, trial)
    vel2 = velocity_trans(joint2, frame, trial)

    # Calculate the velocity of the center of mass of a body part
    vel_com = [(vel1[i] + vel2[i]) / 2 for i in range(3)]
    vel = math.sqrt(sum([v**2 for v in vel_com])) # Magnitude of velocity

    mas = mass(body_part) # Get the mass of the body part

    return 0.5 * mas * vel**2 # Translational kinetic energy formula



def rot_velocity(joint1, joint2, frame, trial):
    '''
    Estimate the rotational velocity between two joints at a given frame and trial.

    Parameters:
    joint1 (str): The name of the first joint.
    joint2 (str): The name of the second joint.
    frame (int): The frame number to evaluate.
    trial (int): The trial number being analyzed.

    Returns:
    float: The estimated rotational velocity in radians per second.
    '''

    pos1 = distance(joint1, joint2, frame, trial) # Distance between joints at the current frame
    pos2 = distance(joint1, joint2, frame + 1, trial) # Distance between joints at the next frame

    # Calculate magnitudes and dot product
    mag1 = math.sqrt(sum([p**2 for p in pos1]))
    mag2 = math.sqrt(sum([p**2 for p in pos2]))
    dot = sum([pos1[i] * pos2[i] for i in range(3)])

    # Calculate the angle of rotation using the dot product
    if mag1 > 0 and mag2 > 0:
        cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
        angle = math.acos(cos_angle)
    else:
        angle = 0

    t_start = time(frame, trial) # Time at the current frame
    t_end = time(frame + 1, trial) # Time at the next frame

    return angle / (t_end - t_start) # Return angular velocity



def rot_inertia(body_part, joint1, joint2, frame, trial):
    '''
    Estimate the moment of inertia for a specific body part based on its geometry.

    Parameters:
    body_part (str): The name of the body part for which to calculate moment of inertia.
    joint1 (str): The name of the first joint defining the body part.
    joint2 (str): The name of the second joint defining the body part.
    frame (int): The frame number to evaluate.
    trial (int): The trial number being analyzed.

    Returns:
    float: The estimated moment of inertia of the specified body part in kg·m².
    '''

    if body_part == 'head':
        # Calculate moment of inertia for the head (modeled as a sphere)
        moment_inertia = 0.4 * mass(body_part) * 0.1**2
    elif body_part == 'trunk':
        # Calculate moment of inertia for the trunk (modeled as a cylinder)
        disp = math.sqrt(sum([i**2 for i in distance(joint1, joint2, frame, trial)]))
        moment_inertia = 0.5 * mass(body_part) * disp**2
    else:
        # Calculate moment of inertia for limbs (modeled as rods)
        disp = math.sqrt(sum([i**2 for i in distance(joint1, joint2, frame, trial)]))
        moment_inertia = 0.33 * mass(body_part) * disp**2

    return moment_inertia # Return moment of inertia



def kinetic_energy_rot(body_part, joint1, joint2, frame, trial):
    '''
    Estimate the rotational kinetic energy of a body part defined by two joints.

    Parameters:
    body_part (str): The name of the body part for which to calculate kinetic energy.
    joint1 (str): The name of the first joint that defines the body part.
    joint2 (str): The name of the second joint that defines the body part.
    frame (int): The frame number to evaluate.
    trial (int): The trial number being analyzed.

    Returns:
    float: The estimated rotational kinetic energy of the specified body part in joules.
    '''

    vel = rot_velocity(joint1, joint2, frame, trial)
    mom_inertia = rot_inertia(body_part, joint1, joint2, frame+1, trial)

    return 0.5 * mom_inertia * vel**2 # Rotational kinetic energy formula



def kinetic_energy_total(body_part, joint1, joint2, frame, trial):
    '''
    Estimate the total kinetic energy (both translational and rotational) of a specific body part.

    Parameters:
    body_part (str): The name of the body part for which to calculate total kinetic energy.
    joint1 (str): The name of the first joint that defines the body part.
    joint2 (str): The name of the second joint that defines the body part.
    frame (int): The frame number to evaluate.
    trial (int): The trial number being analyzed.

    Returns:
    float: The estimated total kinetic energy of the specified body part in joules, 
           combining both translational and rotational components.
    '''

    return kinetic_energy_trans(body_part, joint1, joint2, frame, trial) + kinetic_energy_rot(body_part, joint1, joint2, frame, trial)



def kinetic_energy_efficiency(joint, frame, trial):
    '''
    Estimate the kinetic energy transfer efficiency of a specified joint between frames.

    Parameters:
    joint (str): The name of the joint for which to calculate kinetic energy efficiency.
    frame (int): The frame number to evaluate.
    trial (int): The trial number being analyzed.

    Returns:
    float: The calculated kinetic energy transfer efficiency of the specified joint.
           A value of 1 indicates perfect efficiency, while values less than 1 indicate energy loss.
    '''

    # Define connections between body parts
    # e.g. the right elbow connects the right forearm and the right upper arm
    joints = {'R_ELBOW':[[('R_SHOULDER', 'R_ELBOW'),'upper_arm'],
                     [('R_ELBOW', 'R_WRIST'), 'forearm',]],
        'L_ELBOW': [[('L_SHOULDER', 'L_ELBOW'), 'upper_arm'],
                    [('L_ELBOW', 'L_WRIST'), 'forearm',]],

        'R_KNEE': [[('R_KNEE', 'R_ANKLE'), 'lower_leg'],
                    [('R_HIP', 'R_KNEE'), 'thigh',]],
        'L_KNEE': [[('L_KNEE', 'L_ANKLE'), 'lower_leg'],
                    [('L_HIP', 'L_KNEE'), 'thigh',]],

        'R_HIP': [[('R_HIP', 'R_KNEE'), 'thigh',],
                   [('R_SHOULDER', 'R_HIP'), 'trunk_vertical',]],
        'L_HIP': [[('L_HIP', 'L_KNEE'), 'thigh',],
                  [('L_SHOULDER', 'L_HIP'), 'trunk_vertical',]],

        'R_SHOULDER': [[('R_SHOULDER', 'R_HIP'), 'trunk_vertical',],
                   [('R_SHOULDER', 'R_ELBOW'),'upper_arm']],
        'L_SHOULDER': [[('L_SHOULDER', 'L_HIP'), 'trunk_vertical',],
                  [('L_SHOULDER', 'L_ELBOW'),'upper_arm']],}

    # Calculate the total kinetic energy before and after the frame
    post = kinetic_energy_total(joints[joint][1][1], joints[joint][1][0][0], joints[joint][1][0][1], frame+1, trial)
    pre = kinetic_energy_total(joints[joint][0][1], joints[joint][0][0][0], joints[joint][0][0][1], frame, trial)

    # if the original energy is very small, set the efficiency to 1
    if pre < 1e-20:
        return 1
    return post / pre # Calculate efficiency



def normalize(joint, trial):
    '''
    Normalize the kinetic energy efficiency for a specific joint across frames in a trial.

    Parameters:
    joint (str): The name of the joint for which to normalize kinetic energy efficiency.
    trial (int): The trial number in which the joint's performance is being evaluated.

    Returns:
    tuple: A tuple containing the minimum and maximum kinetic energy efficiency percentiles
           (1st percentile, 99th percentile) for the specified joint and trial.
    '''

    # Calculate kinetic energy efficiency for each frame within the specified trial
    efficiencies = [kinetic_energy_efficiency(joint, frame, trial) for frame in range(43)]

    # Compute the 1st, 50th (median), and 99th percentiles of the efficiencies
    percentiles = np.percentile(sorted(efficiencies), [1, 50, 99])

    return percentiles[0], percentiles[2] # Return the 1st & 99th percentile values



def kinetic_energy_efficiency_norm(joint, frame, trial):
    '''
    Estimate the normalized kinetic energy efficiency of a specific joint for a given frame and trial.

    Parameters:
    joint (str): The name of the joint for which to calculate normalized efficiency.
    frame (int): The frame number in the trial to evaluate.
    trial (int): The trial number being evaluated.

    Returns:
    float: A value between 0 and 1 representing the normalized kinetic energy efficiency
           of the joint for the specified frame and trial.
    '''

    # Calculate the kinetic energy efficiency for the given joint, frame, and trial
    ke_eff = kinetic_energy_efficiency(joint, frame, trial)

    # Small constant to prevent log of zero
    e = 1e-15

    # Retrieve boundaries (1st & 99th percentiles) from the distribution of the data
    mini, maxi = normalize(joint, trial)

    # Normalize the kinetic energy efficiency using the log scale and clip the values between 0 and 1
    return np.clip((np.log10(ke_eff + e) - np.log10(mini + e)) / (np.log10(maxi + e) - np.log10(mini + e)), 0, 1)



def average_kinetic_energy_efficiency(joint, trial):
    '''
    Calculate the average kinetic energy efficiency of a specific joint across frames in a trial.

    Parameters:
    joint (str): The name of the joint for which to calculate the average efficiency.
    trial (int): The trial number in which the joint's average performance is being evaluated.

    Returns:
    float: The average kinetic energy efficiency for the specified joint and trial.
    '''
    
    tot_eff = 0.0 # Initialize total efficiency

    # Set the frame range
    start = 0
    end = 30

    # Sum the kinetic energy efficiencies for the specified frames
    for frame in range(start, end):
        tot_eff += kinetic_energy_efficiency(joint, frame, trial)

    # Return the average efficiency over the frame range
    return tot_eff/30



def relative_energy_contribution(trial):
    '''
    Calculate the relative contributions of the left and right body sides to the total kinetic energy
    during free throw trials.

    Parameters:
    trial (int): The trial number being evaluated.

    Returns:
    float: The percentage contribution of the right side to the total kinetic energy during the trial.
    '''

    # Define mappings for right side body parts and their respective segments
    right = {('R_SHOULDER', 'R_ELBOW'):'upper_arm',
         ('R_ELBOW', 'R_WRIST'): 'forearm',
         ('R_HIP', 'R_KNEE'): 'thigh',
         ('R_KNEE', 'R_ANKLE'): 'lower_leg',}

    # Define mappings for left side body parts and their respective segments
    left = {('L_SHOULDER', 'L_ELBOW'):'upper_arm',
         ('L_ELBOW', 'L_WRIST'): 'forearm',
         ('L_HIP', 'L_KNEE'): 'thigh',
         ('L_KNEE', 'L_ANKLE'): 'lower_leg',}

    # Initialize total energies for both sides of the body
    tot_right = 0.0
    tot_left = 0.0

    # Set the frame range
    start = 0
    end = 30

    # Loop through the specified frames to calculate total energy contributions
    for frame in range(start, end):
        # Sum the kinetic energy for the right side
        for k, v in right.items():
            tot_right += kinetic_energy_total(v, k[0], k[1], frame, trial)
        # Sum the kinetic energy for the left side
        for k, v in left.items():
            tot_left += kinetic_energy_total(v, k[0], k[1], frame, trial)
    tot = tot_right + tot_left

    return tot_right/tot * 100 # Return the percentage contribution of the right side to the total energy



# Create dictionary to map connections to body parts
sequence = {('R_SHOULDER', 'R_ELBOW'):'upper_arm',
            ('L_SHOULDER', 'L_ELBOW'): 'upper_arm',
            ('R_ELBOW', 'R_WRIST'): 'forearm',
            ('L_ELBOW', 'L_WRIST'): 'forearm',

            ('R_HIP', 'R_KNEE'): 'thigh',
            ('L_HIP', 'L_KNEE'): 'thigh',
            ('R_KNEE', 'R_ANKLE'): 'lower_leg',
            ('L_KNEE', 'L_ANKLE'): 'lower_leg',

            ('R_SHOULDER', 'R_HIP'): 'trunk_vertical',
            ('L_SHOULDER', 'L_HIP'): 'trunk_vertical',
            ('R_SHOULDER', 'L_SHOULDER'): 'trunk_horizontal',
            ('R_HIP', 'L_HIP'): 'trunk_horizontal',

            ('R_ANKLE', 'R_CALC'):'hindfoot',
            ('L_ANKLE', 'L_CALC'):'hindfoot',
            ('R_1STTOE', 'R_CALC'):'midfoot',
            ('L_1STTOE', 'L_CALC'):'midfoot',
            ('R_5THTOE', 'R_CALC'):'midfoot',
            ('L_5THTOE', 'L_CALC'):'midfoot',
            ("R_1STTOE", "R_5THTOE"): "forefoot",
            ("L_1STTOE", "L_5THTOE"): "forefoot",

            ("R_WRIST", "R_1STFINGER"): 'hand_thumbside',
            ("R_WRIST", "R_5THFINGER"): 'hand_pinkyside',
            ("L_WRIST", "L_1STFINGER"): 'hand_thumbside',
            ("L_WRIST", "L_5THFINGER"): 'hand_pinkyside',
            ("R_1STFINGER", "R_5THFINGER"): 'hand_palm',
            ("L_1STFINGER", "L_5THFINGER"): 'hand_palm',
            }