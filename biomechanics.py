
##### Functions to calculate Energy Transfers #####

# Import Packages
import math
import json
import numpy as np
from helper import *

# Load Data (get data from every free throw)
data = []

for trial_number in range(1,126):

    trial_id = str(trial_number).zfill(4)

    with open(f'./data/P0001/BB_FT_P0001_T{trial_id}.json') as json_file:
        free_throw_data = json.load(json_file)
        data.append(free_throw_data)

# Grab only data from around the actual shot (30 second before ball release to 15 seconds after)
data = align_tracking_data(data)

# Function to get the position (in metres) of a point on the body at a given frame and trial (x, y, z positions)
def position(joint, frame, trial):
    return [p / 3.281 for p in data[trial]['tracking'][frame]['data']['player'][joint]] # Convert feet to metres

# Function to get the time (in s) associated with a particular frame in a trial
def time(frame, trial):
    return data[trial]['tracking'][frame]['time'] / 1000 # Convert milliseconds to seconds

# Function to get the distance between two joints on the body at a given frame and trial (x, y, z distances)
def distance(joint1, joint2, frame, trial):
    return [p1 - p2 for p1, p2 in zip(position(joint1, frame, trial), position(joint2, frame, trial))]

# Function to estimate velocity of a point on the body (x, y, z velocities)
def velocity_trans(joint, frame, trial):
    pos_start = position(joint, frame, trial) # Position at the current frame
    pos_end = position(joint, frame + 1, trial) # Position at the next frame

    t_start = time(frame, trial) # Time at the current frame
    t_end = time(frame + 1, trial) # Time at the next frame

    return [((pos_end[i] - pos_start[i]) / (t_end - t_start)) for i in range(3)] # Calculate velocity

# Function to estimate the mass of a body part
def mass(body_part):
    # Estimate the total mass using an average adult male in Canada
    total_mass = 90
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

# Function to estimate translational kinetic energy for a body part between two joints
def kinetic_energy_trans(body_part, joint1, joint2, frame, trial):

    # Calculate the velocity of each connecting joint
    vel1 = velocity_trans(joint1, frame, trial)
    vel2 = velocity_trans(joint2, frame, trial)

    # Calculate the velocity of the center of mass of a body part
    vel_com = [(vel1[i] + vel2[i]) / 2 for i in range(3)]
    vel = math.sqrt(sum([v**2 for v in vel_com])) # Magnitude of velocity

    mas = mass(body_part) # Get the mass of the body part

    return 0.5 * mas * vel**2 # Translational kinetic energy formula

# Function to estimate the rotational velocity of a body part between two joints
def rot_velocity(joint1, joint2, frame, trial):

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

# Function to estimate the moment of inertia of a body part
def rot_inertia(body_part, joint1, joint2, frame, trial):
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

# Function to estimate rotational kinetic energy of a body part
def kinetic_energy_rot(body_part, joint1, joint2, frame, trial):
    vel = rot_velocity(joint1, joint2, frame, trial)

    # vel = math.sqrt(sum([v**2 for v in vel]))

    mom_inertia = rot_inertia(body_part, joint1, joint2, frame+1, trial)

    return 0.5 * mom_inertia * vel**2

# Function to estimate the total kinetic energy of a body part
def kinetic_energy_total(body_part, joint1, joint2, frame, trial):
    return kinetic_energy_trans(body_part, joint1, joint2, frame, trial) + kinetic_energy_rot(body_part, joint1, joint2, frame, trial)

# Function to estimate the change in kinetic energy of a joint
def kinetic_energy_efficiency(joint, frame, trial):

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

    post = kinetic_energy_total(joints[joint][1][1], joints[joint][1][0][0], joints[joint][1][0][1], frame+1, trial)
    pre = kinetic_energy_total(joints[joint][0][1], joints[joint][0][0][0], joints[joint][0][0][1], frame, trial)

    # if the original energy is very small, set the efficiency to 1
    if pre < 1e-12:
        return 1
    return post / pre

# Function to normalize kinetic energy efficiency for each joint
def normalize(joint, trial):
    efficiencies = [kinetic_energy_efficiency(joint, frame, trial) for frame in range(239)]
    percentiles = np.percentile(sorted(efficiencies), [1, 50, 99])
    return percentiles[0], percentiles[2]

# Function to estimate the normalized kinetic energy efficiency of a joint
def kinetic_energy_efficiency_norm(joint, frame, trial):
    ke_eff = kinetic_energy_efficiency(joint, frame, trial)
    e = 1e-15
    # Retrieve boundaries from the distribution of the data
    mini, maxi = normalize(joint, trial)
    return np.clip((np.log10(ke_eff + e) - np.log10(mini + e)) / (np.log10(maxi + e) - np.log10(mini + e)), 0, 1)

# Function to calculate the average kinetic energy efficiency for a particular joint for a particular trial
def average_kinetic_energy_efficiency(joint, trial):
    
    tot_eff = 0.0

    start = 0
    end = 30

    for frame in range(start, end):
        tot_eff += kinetic_energy_efficiency(joint, frame, trial)
    return tot_eff/30

# Function to calculate the relative contributions of the left and right sides to free throws
def relative_energy_contribution(trial):

    right = {('R_SHOULDER', 'R_ELBOW'):'upper_arm',
         ('R_ELBOW', 'R_WRIST'): 'forearm',
         ('R_HIP', 'R_KNEE'): 'thigh',
         ('R_KNEE', 'R_ANKLE'): 'lower_leg',}

    left = {('L_SHOULDER', 'L_ELBOW'):'upper_arm',
         ('L_ELBOW', 'L_WRIST'): 'forearm',
         ('L_HIP', 'L_KNEE'): 'thigh',
         ('L_KNEE', 'L_ANKLE'): 'lower_leg',}

    tot_right = 0.0
    tot_left = 0.0

    start = 0
    end = 30
    for frame in range(start, end):
        for k, v in right.items():
            tot_right += kinetic_energy_total(v, k[0], k[1], frame, trial)
        for k, v in left.items():
            tot_left += kinetic_energy_total(v, k[0], k[1], frame, trial)
    tot = tot_right + tot_left

    return tot_right/tot * 100

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