# -*- coding: utf-8 -*-
"""Build future lane-change prediction samples from the official highD data.

Each sample contains 50 historical frames (2 s at 25 Hz). Its label is based
on whether the first lane-ID transition in the next 75 frames (3 s) is keep,
left, or right. Candidate prediction times are sampled every 25 frames (1 s)
to keep preprocessing and CPU training tractable.

Output format remains compatible with the project's rnn_model.py:
    [(numpy_array_of_shape_50x16, label), ...]
Labels: 0=keep, 1=left, 2=right.
"""

from read_data import *
import os
import math
import random
import numpy as np

FRAME_TAKEN = 50                    # 50 history frames = 2 seconds at 25 Hz
FUTURE_SECONDS = 3                  # prediction horizon
SAMPLE_STRIDE = 25                  # one candidate time every 25 frames = 1 second
KEEP_TO_CHANGE_RATIO = 3.0          # downsample keep candidates to 3x positives per recording
MAX_KEEP_CANDIDATES_PER_RECORDING = 5000  # memory/runtime safeguard
MIN_KEEP_SAMPLES_PER_RECORDING = 100
RANDOM_SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run(number):
    '''
    This function runs the data processing code and output to pickle files
    '''
    # Read the three official highD files using paths relative to this script.
    data_dir = os.path.join(BASE_DIR, "data")
    tracks_csv = read_tracks_csv(os.path.join(data_dir, number + "_tracks.csv"))
    tracks_meta = read_tracks_meta(os.path.join(data_dir, number + "_tracksMeta.csv"))
    recording_meta = read_recording_meta(
        os.path.join(data_dir, number + "_recordingMeta.csv"))

    # figure out the lane changing cars and lane keeping cars
    lane_changing_ids = []
    lane_keeping_ids = []
    for key in tracks_meta:
        if(tracks_meta[key][NUMBER_LANE_CHANGES] > 0):
            lane_changing_ids.append(key)
        else:
            lane_keeping_ids.append(key)

    # get the lane information
    lanes_info = {}
    lane_num = len(recording_meta[UPPER_LANE_MARKINGS]) + \
        len(recording_meta[LOWER_LANE_MARKINGS]) - 2
    if lane_num == 4:
        # 4 lanes
        lanes_info[2] = recording_meta[UPPER_LANE_MARKINGS][0]
        lanes_info[3] = recording_meta[UPPER_LANE_MARKINGS][1]
        lanes_info[5] = recording_meta[LOWER_LANE_MARKINGS][0]
        lanes_info[6] = recording_meta[LOWER_LANE_MARKINGS][1]
        lane_width = ((lanes_info[3] - lanes_info[2]) +
                      (lanes_info[6] - lanes_info[5])) / 2
    elif lane_num == 6:
        # 6 lanes
        lanes_info[2] = recording_meta[UPPER_LANE_MARKINGS][0]
        lanes_info[3] = recording_meta[UPPER_LANE_MARKINGS][1]
        lanes_info[4] = recording_meta[UPPER_LANE_MARKINGS][2]
        lanes_info[6] = recording_meta[LOWER_LANE_MARKINGS][0]
        lanes_info[7] = recording_meta[LOWER_LANE_MARKINGS][1]
        lanes_info[8] = recording_meta[LOWER_LANE_MARKINGS][2]
        lane_width = ((lanes_info[3] - lanes_info[2]) + (lanes_info[4] - lanes_info[3]) +
                      (lanes_info[7] - lanes_info[6]) + (lanes_info[8] - lanes_info[7])) / 4
    elif lane_num == 7:
        # 7 lanes: track 58 ~ 60
        lanes_info[2] = recording_meta[UPPER_LANE_MARKINGS][0]
        lanes_info[3] = recording_meta[UPPER_LANE_MARKINGS][1]
        lanes_info[4] = recording_meta[UPPER_LANE_MARKINGS][2]
        lanes_info[5] = recording_meta[UPPER_LANE_MARKINGS][3]
        lanes_info[7] = recording_meta[LOWER_LANE_MARKINGS][0]
        lanes_info[8] = recording_meta[LOWER_LANE_MARKINGS][1]
        lanes_info[9] = recording_meta[LOWER_LANE_MARKINGS][2]
        lane_width = ((lanes_info[3] - lanes_info[2]) + (lanes_info[4] - lanes_info[3]) + (
            lanes_info[5] - lanes_info[4]) + (lanes_info[8] - lanes_info[7]) + (lanes_info[9] - lanes_info[8])) / 5
    else:
        print("Error: Invalid input -", number)

    def determine_lane_exist(cur_lane):
        '''
        return: left_exist, right_exist 
        Have to do this shit in a hardcoded way to determine the existence of neighbor lanes.
        '''
        if lane_num == 4:
            if cur_lane == 2 or cur_lane == 6:
                return 1, 0
            else:
                return 0, 1
        elif lane_num == 6:
            if cur_lane == 2 or cur_lane == 8:
                return 1, 0
            elif cur_lane == 3 or cur_lane == 7:
                return 1, 1
            else:
                return 0, 1
        elif lane_num == 7:
            if cur_lane == 2 or cur_lane == 9:
                return 1, 0
            elif cur_lane == 3 or cur_lane == 4 or cur_lane == 8:
                return 1, 1
            else:
                return 0, 1

    def construct_features(i, frame_num, original_lane):
        '''
        Construct all the features for the RNN to train:
        Here is the list:
        Difference of the ego car’s Y position and the lane center: ΔY
        Ego car’s X velocity: Vx
        Ego car’s Y velocity: Vy
        Ego car’s X acceleration: Ax
        Ego car’s Y acceleration: Ay
        Ego car type: T
        TTC of preceding car: TTCp
        TTC of following car: TTCf
        TTC of left preceding car: TTClp
        TTC of left alongside car: TTCla
        TTC of left following car: TTClf
        TTC of right preceding car: TTCrp
        TTC of right alongside car: TTCra
        TTC of right following car: TTCrf
        '''
        going = 0  # 1 left, 2 right
        if lane_num == 4:
            if original_lane == 2 or original_lane == 3:
                going = 1
            else:
                going = 2
        else:
            if original_lane == 2 or original_lane == 3 or original_lane == 4 or original_lane == 5:
                going = 1
            else:
                going = 2
        cur_feature = {}
        cur_feature["left_lane_exist"], cur_feature["right_lane_exist"] = determine_lane_exist(
            original_lane)

        # We need to consider the fact that right/left are different for top/bottom lanes.
        # top lanes are going left      <----
        # bottom lanes are going right  ---->
        # left -> negative, right -> positive
        if going == 1:
            cur_feature["delta_y"] = tracks_csv[i][Y][frame_num] - \
                lanes_info[original_lane]  # up
            cur_feature["y_velocity"] = -tracks_csv[i][Y_VELOCITY][frame_num]
            cur_feature["y_acceleration"] = - \
                tracks_csv[i][Y_ACCELERATION][frame_num]
        else:
            cur_feature["delta_y"] = lanes_info[original_lane] - \
                tracks_csv[i][Y][frame_num]  # down
            cur_feature["y_velocity"] = tracks_csv[i][Y_VELOCITY][frame_num]
            cur_feature["y_acceleration"] = tracks_csv[i][Y_ACCELERATION][frame_num]

        cur_feature["x_velocity"] = tracks_csv[i][X_VELOCITY][frame_num]
        cur_feature["x_acceleration"] = tracks_csv[i][X_ACCELERATION][frame_num]
        cur_feature["car_type"] = 1 if tracks_meta[i][CLASS] == "Car" else -1

        def calculate_ttc(target_car_id):
            """Calculate TTC; use a large sentinel if a neighbor is unavailable."""
            if target_car_id == 0 or target_car_id not in tracks_csv or target_car_id not in tracks_meta:
                return 99999
            try:
                target_frame = int(
                    tracks_meta[i][INITIAL_FRAME] + frame_num
                    - tracks_meta[target_car_id][INITIAL_FRAME]
                )
                if target_frame < 0 or target_frame >= len(tracks_csv[target_car_id][X]):
                    return 99999
                target_x = tracks_csv[target_car_id][X][target_frame]
                cur_x = tracks_csv[i][X][frame_num]
                target_v = tracks_csv[target_car_id][X_VELOCITY][target_frame]
                cur_v = tracks_csv[i][X_VELOCITY][frame_num]
                if abs(float(target_v) - float(cur_v)) < 1e-8:
                    return 99999
                if going == 1:
                    if cur_x > target_x:
                        denominator = cur_v - target_v
                        ttc = (cur_x - target_x) / denominator
                    else:
                        denominator = target_v - cur_v
                        ttc = (target_x - cur_x) / denominator
                else:
                    if cur_x > target_x:
                        denominator = target_v - cur_v
                        ttc = (cur_x - target_x) / denominator
                    else:
                        denominator = cur_v - target_v
                        ttc = (target_x - cur_x) / denominator
                if not np.isfinite(ttc) or ttc < 0:
                    return 99999
                return float(ttc)
            except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
                return 99999

        # surrounding cars info
        cur_feature["preceding_ttc"] = calculate_ttc(
            tracks_csv[i][PRECEDING_ID][frame_num])
        cur_feature["following_ttc"] = calculate_ttc(
            tracks_csv[i][FOLLOWING_ID][frame_num])
        cur_feature["left_preceding_ttc"] = calculate_ttc(
            tracks_csv[i][LEFT_PRECEDING_ID][frame_num])
        cur_feature["left_alongside_ttc"] = calculate_ttc(
            tracks_csv[i][LEFT_ALONGSIDE_ID][frame_num])
        cur_feature["left_following_ttc"] = calculate_ttc(
            tracks_csv[i][LEFT_FOLLOWING_ID][frame_num])
        cur_feature["right_preceding_ttc"] = calculate_ttc(
            tracks_csv[i][RIGHT_PRECEDING_ID][frame_num])
        cur_feature["right_alongside_ttc"] = calculate_ttc(
            tracks_csv[i][RIGHT_ALONGSIDE_ID][frame_num])
        cur_feature["right_following_ttc"] = calculate_ttc(
            tracks_csv[i][RIGHT_FOLLOWING_ID][frame_num])

        ret = tuple(cur_feature.values())
        return ret

    def detect_lane_change(lane_center, cur_y, lane_width, car_height):
        delta_y = abs(lane_center - cur_y)
        relative_diff = delta_y / car_height
        if(relative_diff < 0.5):
            return True
        else:
            return False

    def determine_change_direction(ori_laneId, new_laneId):
        '''
        return 1 upon left change
        return 2 upon right change
        '''
        if lane_num == 4:
            if (ori_laneId == 2 and new_laneId == 3) or (ori_laneId == 6 and new_laneId == 5):
                return 1
            else:
                return 2
        else:
            # left:
            if (ori_laneId == 2 and new_laneId == 3) or (ori_laneId == 4 and new_laneId == 5) \
                or (ori_laneId == 3 and new_laneId == 4) or (ori_laneId == 7 and new_laneId == 6) \
                    or (ori_laneId == 8 and new_laneId == 7) or (ori_laneId == 9 and new_laneId == 8):
                return 1
            else:
                return 2

    # -------------------------------------------------------------------------
    # Build candidate-time samples for the actual future-prediction task.
    # A candidate is valid only when:
    #   * 50 historical frames and 75 future frames are available;
    #   * all 125 frames are consecutive in the official frame index;
    #   * the lane ID is stable throughout the 50-frame input history.
    # The label is determined only from the future 75 frames.
    # -------------------------------------------------------------------------
    recording_id = int(number)
    rng = random.Random(RANDOM_SEED + recording_id)
    future_frames = int(round(recording_meta[FRAME_RATE] * FUTURE_SECONDS))
    if future_frames <= 0:
        raise ValueError("Invalid frame rate in recording metadata: {}".format(
            recording_meta[FRAME_RATE]))

    # Candidate format: (track_id, start_idx, current_idx, current_lane, label)
    positive_candidates = []
    keep_reservoir = []
    keep_seen = 0

    for track_id, track in tracks_csv.items():
        n_frames = len(track[FRAME])
        if n_frames < FRAME_TAKEN + future_frames:
            continue

        frame_numbers = np.asarray(track[FRAME])
        lane_ids = np.asarray(track[LANE_ID])

        # t is the final frame of the historical input window.
        for t in range(FRAME_TAKEN - 1, n_frames - future_frames, SAMPLE_STRIDE):
            start_idx = t - FRAME_TAKEN + 1
            stop_future = t + future_frames  # inclusive future endpoint

            # Skip gaps so that 50 history + 75 future frames really mean 5 seconds.
            window_frames = frame_numbers[start_idx:stop_future + 1]
            if len(window_frames) != FRAME_TAKEN + future_frames:
                continue
            if np.any(np.diff(window_frames) != 1):
                continue

            current_lane = int(lane_ids[t])
            if current_lane not in lanes_info:
                continue

            # The observed history must precede the next lane change, not include
            # a lane transition that has already happened.
            history_lanes = lane_ids[start_idx:t + 1]
            if len(history_lanes) != FRAME_TAKEN or not np.all(history_lanes == current_lane):
                continue

            # Label from the FIRST lane-ID transition strictly after t and within
            # the next FUTURE_SECONDS. None means an unusable/unknown transition.
            label = 0
            invalid_transition = False
            for future_idx in range(t + 1, stop_future + 1):
                previous_lane = int(lane_ids[future_idx - 1])
                next_lane = int(lane_ids[future_idx])
                if next_lane != previous_lane:
                    if previous_lane not in lanes_info or next_lane not in lanes_info:
                        invalid_transition = True
                    else:
                        label = determine_change_direction(previous_lane, next_lane)
                    break
            if invalid_transition:
                continue

            candidate = (track_id, start_idx, t, current_lane, int(label))
            if label in (1, 2):
                positive_candidates.append(candidate)
            else:
                # Reservoir sampling limits memory while keeping a random sample
                # of the many more abundant keep candidates.
                keep_seen += 1
                if len(keep_reservoir) < MAX_KEEP_CANDIDATES_PER_RECORDING:
                    keep_reservoir.append(candidate)
                else:
                    replace_idx = rng.randrange(keep_seen)
                    if replace_idx < MAX_KEEP_CANDIDATES_PER_RECORDING:
                        keep_reservoir[replace_idx] = candidate

    # Keep all positive candidates and sample a controlled number of keep examples.
    # This avoids producing millions of near-duplicate negative windows from highD.
    if positive_candidates:
        desired_keep = int(math.ceil(len(positive_candidates) * KEEP_TO_CHANGE_RATIO))
    else:
        desired_keep = MIN_KEEP_SAMPLES_PER_RECORDING
    desired_keep = min(desired_keep, len(keep_reservoir))
    selected_keep = (
        rng.sample(keep_reservoir, desired_keep)
        if desired_keep < len(keep_reservoir)
        else list(keep_reservoir)
    )
    selected_candidates = positive_candidates + selected_keep
    rng.shuffle(selected_candidates)

    result = []
    failed_samples = 0
    for track_id, start_idx, t, current_lane, label in selected_candidates:
        try:
            sequence = [
                construct_features(track_id, frame_idx, current_lane)
                for frame_idx in range(start_idx, t + 1)
            ]
            sequence = np.asarray(sequence, dtype=np.float32)
            if sequence.shape != (FRAME_TAKEN, 16):
                failed_samples += 1
                continue
            # Keep the model input finite; unavailable TTC values use 99999 above.
            sequence = np.nan_to_num(sequence, nan=0.0, posinf=99999.0, neginf=-99999.0)
            result.append((sequence, int(label)))
        except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
            failed_samples += 1
            continue

    # `change_num` now means the number of positive prediction windows, not the
    # number of unique physical lane-change events.
    change_num = sum(1 for _, label in result if label in (1, 2))
    counts = {0: 0, 1: 0, 2: 0}
    for _, label in result:
        counts[int(label)] += 1
    print(
        "Recording {}: candidates={} | keep={} left={} right={} | skipped_feature_samples={}".format(
            number, len(result), counts[0], counts[1], counts[2], failed_samples
        )
    )
    return result, change_num
