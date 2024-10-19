

# Helper Functions
import math
import json

# Load Data (get data from every free throw)
data = []

for trial_number in range(1,126):

    trial_id = str(trial_number).zfill(4)

    with open(f'./data/P0001/BB_FT_P0001_T{trial_id}.json') as json_file:
        free_throw_data = json.load(json_file)
        data.append(free_throw_data)

# Function to find the frame
def ball_leaves_hands(trial):
    
    locations = [data[trial]['tracking'][i]['data']['ball'] for i in range(len(data[trial]['tracking']))]
    ball_frame = None
    for i, loc in enumerate(reversed(locations)):
        if not all(math.isnan(l) for l in loc):
            ball_frame = len(locations) - 1 - i
            break
    
    search_range = range(ball_frame - 15, ball_frame)

    for frame in search_range:
        ball_pos = data[trial]['tracking'][frame]['data']['ball']
        
        finger_pos = data[trial]['tracking'][frame]['data']['player']['R_1STFINGER']
        
        ball_finger_dist = math.sqrt(sum([(p1 - p2)**2 for p1, p2 in zip(ball_pos, finger_pos)]))
        
        if ball_finger_dist > 0.5:
            return frame

# Get the time that the ball leaves the player's hands for each trial

leaves_hands = []

for trial in range(0, 125):
    leaves_hands.append(ball_leaves_hands(trial))


# Function to align trials based on when the ball leaves their hands
def align_tracking_data(trials, num_frames_back=30, num_frames_forward=15):
    aligned_trials = []

    for i, trial in enumerate(trials):
        participant_id = trial['participant_id']
        trial_id = trial['trial_id']
        landing_x = trial['landing_x']
        landing_y = trial['landing_y']
        entry_angle = trial['entry_angle']

        aligned_frame = leaves_hands[i]

        # Collect tracking data for the aligned frame and the 30 frames back
        tracking_data_range = []
        for frame in range(aligned_frame - num_frames_back, aligned_frame + num_frames_forward):
            frame_data = next(
                (t['data'] for t in trial['tracking'] if t['frame'] == frame), None)

            if frame_data:
                reset_frame = frame - (aligned_frame - num_frames_back)
                tracking_data_range.append({
                    'frame': reset_frame,
                    'time': reset_frame*33,  # Assuming 'time' needs to be taken from the tracking data
                    'data': frame_data
                })

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

    return aligned_trials