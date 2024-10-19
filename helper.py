

##### Helper Functions #####

# Import Packages
import math
import json



# Load Data (get data from every free throw)
data = []

# Loop through trial numbers 1 to 125 and load corresponding JSON files
for trial_number in range(1,126):
    trial_id = str(trial_number).zfill(4) # Format trial number with leading zeros

    # Open and load JSON data for the current trial
    with open(f'./data/P0001/BB_FT_P0001_T{trial_id}.json') as json_file:
        free_throw_data = json.load(json_file)
        data.append(free_throw_data)



def ball_leaves_hands(trial):
    '''
    Determine the frame in which the ball leaves the player's hands for a specific trial.

    Parameters:
    trial (int): The index of the trial in the loaded data.

    Returns:
    int: The frame number when the ball leaves the player's hands.
         If the frame cannot be determined, returns None.
    '''
    
    # Extract ball tracking locations from the trial data
    locations = [data[trial]['tracking'][i]['data']['ball'] for i in range(len(data[trial]['tracking']))]
    ball_frame = None

    # Find the last frame where the ball has valid tracking data
    for i, loc in enumerate(reversed(locations)):
        if not all(math.isnan(l) for l in loc):
            ball_frame = len(locations) - 1 - i
            break
    
    # Define a search range around the ball frame to check for hand release
    search_range = range(ball_frame - 15, ball_frame)

    # Iterate through the search range to find the release frame
    for frame in search_range:
        ball_pos = data[trial]['tracking'][frame]['data']['ball']
        finger_pos = data[trial]['tracking'][frame]['data']['player']['R_1STFINGER']
        
        # Calculate the distance between the ball and the index finger
        ball_finger_dist = math.sqrt(sum([(p1 - p2)**2 for p1, p2 in zip(ball_pos, finger_pos)]))
        
        # If the distance exceeds a threshold (half a foot), return the frame number
        if ball_finger_dist > 0.5:
            return frame



# Get the time that the ball leaves the player's hands for each trial
leaves_hands = []
for trial in range(0, 125):
    leaves_hands.append(ball_leaves_hands(trial))



def align_tracking_data(trials, num_frames_back=30, num_frames_forward=15):
    '''
    Align tracking data for multiple trials based on the frame when the ball leaves the player's hands.

    Parameters:
    trials (list): A list of trial data, each containing tracking information.
    num_frames_back (int): The number of frames to include before the ball leaves hands.
    num_frames_forward (int): The number of frames to include after the ball leaves hands.

    Returns:
    list: A list of aligned trial data, including tracking information.
    '''

    aligned_trials = []

    # Iterate through each trial to align tracking data
    for i, trial in enumerate(trials):
        participant_id = trial['participant_id']
        trial_id = trial['trial_id']
        landing_x = trial['landing_x']
        landing_y = trial['landing_y']
        entry_angle = trial['entry_angle']

        # Get the aligned frame based on when the ball leaves hands
        aligned_frame = leaves_hands[i]

        # Collect tracking data for the aligned frame and the specified number of frames back and forward
        tracking_data_range = []
        for frame in range(aligned_frame - num_frames_back, aligned_frame + num_frames_forward):
            # Find the tracking data for the current frame
            frame_data = next(
                (t['data'] for t in trial['tracking'] if t['frame'] == frame), None)

            if frame_data:
                reset_frame = frame - (aligned_frame - num_frames_back) # Reset frame number for alignment
                tracking_data_range.append({
                    'frame': reset_frame,
                    'time': reset_frame*33,  # Assuming 33 ms per frame
                    'data': frame_data
                })

        # If tracking data exists, create an aligned trial dictionary
        if tracking_data_range:
            aligned_trial = {
                'participant_id': participant_id,
                'trial_id': trial_id,
                'result': trial['result'],
                'landing_x': landing_x,
                'landing_y': landing_y,
                'entry_angle': entry_angle,
                'tracking': tracking_data_range
            }
            aligned_trials.append(aligned_trial)

    return aligned_trials # Return the list of aligned trials