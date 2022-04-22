import os

from argoverse.map_representation.map_api import ArgoverseMap
from mai.utils import FI
from mai.utils import io
import numpy as np
from tqdm import tqdm
import pandas as pd
import time
import multiprocessing as mp
from functools import partial


@FI.register
class ArgoPredSummary(object):
    def __init__(self):
        super().__init__()

    def __call__(self, root_path, split, **kwargs):
        summary(root_path, split, **kwargs)


def summary(root_path, split, lane_size=16, lane_reso=1.0, num_workers=1):
    summary_map_new(root_path, lane_size, lane_reso)
    summary_traj(root_path, split, num_workers)


def summary_data(root_path, split, am, obs_len, obj_radius, lane_radius, valid_len, aug, num_workers):
    data_path = os.path.join(root_path, split, 'data')
    frame_name_list = list(os.listdir(data_path))

    # split task
    frame_name_list_split = []
    split_size = int(len(frame_name_list)/num_workers)
    for i in range(num_workers):
        if i == num_workers-1:
            frame_name_list_split.append(frame_name_list[i*split_size:])
        else:
            frame_name_list_split.append(
                frame_name_list[i*split_size:(i+1)*split_size])

    # process using multiprocessing
    with mp.Pool(num_workers) as p:
        results = p.map(partial(summary_list, root_path=root_path, split=split,
                        am=am, obs_len=obs_len, obj_radius=obj_radius, lane_radius=lane_radius, valid_len=valid_len, aug=aug), frame_name_list_split)

    info_list = []
    for res in results:
        info_list.extend(res)
        del res  # release memory

    io.dump(info_list, os.path.join(root_path, f'{split}_info.pkl'))


def summary_list(frame_name_list, root_path, split, am, obs_len, obj_radius, lane_radius, valid_len, aug):
    info_list = []
    for frame_name in tqdm(frame_name_list):
        frame_path = os.path.join(root_path, split, 'data', frame_name)
        df = pd.read_csv(frame_path)
        name, ext = os.path.splitext(frame_name)
        info_list.extend(process_dataframe(
            df, am, name, obs_len=obs_len, obj_radius=obj_radius, lane_radius=lane_radius, valid_len=valid_len, aug=aug))
    return info_list


def summary_map(root_path):
    am = ArgoverseMap()
    lane_dict = am.build_centerline_index()

    # go through each lane segment
    dict = {'PIT': [], 'MIA': []}
    lane_id2idx = {'PIT': {}, 'MIA': {}}
    for city_name in ['PIT', 'MIA']:
        for i, lane_id in tqdm(enumerate(lane_dict[city_name].keys())):
            lane_cl = am.get_lane_segment_centerline(lane_id, city_name)
            is_intersection = am.lane_is_in_intersection(lane_id, city_name)
            turn_direction = am.get_lane_turn_direction(lane_id, city_name)
            traffic_control = am.lane_has_traffic_control_measure(
                lane_id, city_name)
            lane_info1 = 1
            if(is_intersection):
                lane_info1 = 2
            lane_info2 = 1
            if(turn_direction == "LEFT"):
                lane_info2 = 2
            elif(turn_direction == "RIGHT"):
                lane_info2 = 3
            lane_info3 = 1
            if(traffic_control):
                lane_info3 = 2

            # there 61 lane is not enough for size 10, pad on tail
            if len(lane_cl) < 10:
                lane_cl = np.pad(
                    lane_cl, ((0, 10-len(lane_cl)), (0, 0)), mode='edge')

            lane_nd = np.concatenate([lane_cl[:, :2],
                                      np.full((len(lane_cl), 1), lane_info1),
                                      np.full((len(lane_cl), 1), lane_info2),
                                      np.full((len(lane_cl), 1), lane_info3)], axis=-1)
            dict[city_name].append(lane_nd)
            lane_id2idx[city_name][lane_id] = i

        dict[city_name] = np.stack(dict[city_name], axis=0).astype(np.float32)

    io.dump([dict, lane_id2idx], os.path.join(root_path, 'map_info.pkl'))
    for city, lane in dict.items():
        print(f'{city}: {len(lane)} lanes')

    return am


def summary_traj(root_path, split, num_workers=1):
    data_path = os.path.join(root_path, split, 'data')
    frame_name_list = list(os.listdir(data_path))

    offset = 0
    traj_list = []
    info_list = []
    with mp.Pool(num_workers) as p:
        for i, (traj, city) in tqdm(enumerate(
            p.imap(partial(summary_frame_traj,
                           root_path=root_path,
                           split=split),
                   frame_name_list)
        ), total=len(frame_name_list)):
            name, _ = os.path.splitext(frame_name_list[i])
            id = int(name)
            traj_list.append(traj)
            info = dict(
                traj_index=(offset, offset+len(traj)),
                id=id,
                city=city,
            )
            info_list.append(info)
            offset += len(traj)
    total_traj = np.concatenate(traj_list, axis=0)

    np.save(os.path.join(root_path, f'{split}_traj.npy'), total_traj)
    io.dump(info_list, os.path.join(root_path, f'{split}_info.pkl'))


def summary_frame_traj(fname, root_path, split):
    frame_path = os.path.join(root_path, split, 'data', fname)
    df = pd.read_csv(frame_path)

    city_name = df['CITY_NAME'].iloc[0]

    df['TIMESTAMP'] -= np.min(df['TIMESTAMP'].values)
    seq_ts = np.unique(df['TIMESTAMP'].values)
    ts2idx = {ts: i for i, ts in enumerate(seq_ts)}

    # agent
    agent_traj = np.zeros((len(seq_ts), 4), dtype=np.float32)
    for obj_type, obj_df in df.groupby('OBJECT_TYPE'):
        if obj_type == 'AGENT':
            obj_cord = obj_df[['X', 'Y']].values
            obj_ts = obj_df[['TIMESTAMP']].values.squeeze(1).tolist()
            for ts, cord in zip(obj_ts, obj_cord):
                idx = ts2idx[ts]
                agent_traj[idx, :2] = cord
                agent_traj[idx, 2] = ts
                agent_traj[idx, 3] = 1.0

    agent_traj_list = [agent_traj]

    # obj
    for _, obj_df in df.groupby('TRACK_ID'):
        if obj_df['OBJECT_TYPE'].iloc[0] == 'AGENT':  # skip agent itself
            continue

        obj_traj = np.zeros((len(seq_ts), 4), dtype=np.float32)
        obj_traj[:, 2] = agent_traj[:, 2]  # timestamp
        obj_cord = obj_df[['X', 'Y']].values
        obj_ts = obj_df[['TIMESTAMP']].values.squeeze(1).tolist()
        for ts, cord in zip(obj_ts, obj_cord):
            idx = ts2idx[ts]
            obj_traj[idx, :2] = cord
            obj_traj[idx, 3] = 1.0
        agent_traj_list.append(obj_traj)

        # do pad
        start_idx = ts2idx[obj_ts[0]]   # include
        end_idx = ts2idx[obj_ts[-1]]+1  # exclude
        obj_traj[:start_idx, :2] = obj_traj[start_idx, :2]  # left
        obj_traj[end_idx:, :2] = obj_traj[end_idx-1, :2]  # right

    agent_traj_np = np.stack(agent_traj_list, axis=0)

    return agent_traj_np, city_name


def summary_map_new(root_path, lane_size, lane_reso):
    am = ArgoverseMap()
    lane_dict = am.build_centerline_index()

    # go through each lane segment
    dict = {'PIT': [], 'MIA': []}
    lane_id2idx = {'PIT': {}, 'MIA': {}}
    for city_name in ['PIT', 'MIA']:
        for i, lane_id in tqdm(enumerate(lane_dict[city_name].keys())):
            lane_cl = am.get_lane_segment_centerline(lane_id, city_name)[:, :2]
            is_intersection = am.lane_is_in_intersection(lane_id, city_name)
            turn_direction = am.get_lane_turn_direction(lane_id, city_name)
            traffic_control = am.lane_has_traffic_control_measure(
                lane_id, city_name)
            lane_info1 = 1
            if(is_intersection):
                lane_info1 = 2
            lane_info2 = 1
            if(turn_direction == "LEFT"):
                lane_info2 = 2
            elif(turn_direction == "RIGHT"):
                lane_info2 = 3
            lane_info3 = 1
            if(traffic_control):
                lane_info3 = 2

            lane_attri = np.concatenate([np.full((lane_size, 1), lane_info1),
                                         np.full((lane_size, 1), lane_info2),
                                         np.full((lane_size, 1), lane_info3)], axis=-1)

            lane_list = []
            p_last = lane_cl[0]
            lane = [p_last]
            s = 0
            for j in range(1, len(lane_cl)):
                p_curr = lane_cl[j]
                delta = p_curr - p_last
                dist = np.linalg.norm(delta)
                length = 0
                while s+dist-length >= lane_reso:
                    length += lane_reso - s
                    p_sample = length/dist * delta + p_last
                    lane.append(p_sample)
                    if len(lane) == lane_size:
                        lane_np = np.stack(lane, axis=0)
                        lane_np = np.concatenate(
                            [lane_np, lane_attri], axis=-1)
                        lane_list.append(lane_np)
                        lane = [p_sample]
                    s = 0
                s += dist-length
                p_last = p_curr

                if j == len(lane_cl)-1:
                    lane.append(p_last)
                    lane_np = np.stack(lane, axis=0)
                    if len(lane_np) < lane_size:  # pad to lane_size
                        lane_np = np.pad(
                            lane_np, ((0, lane_size-len(lane_np)), (0, 0)), mode='edge')
                    lane_np = np.concatenate([lane_np, lane_attri], axis=-1)
                    lane_list.append(lane_np)

            lane_id2idx[city_name][lane_id] = list(
                range(len(dict[city_name]), len(dict[city_name])+len(lane_list)))
            dict[city_name].extend(lane_list)

        dict[city_name] = np.stack(dict[city_name], axis=0).astype(np.float32)

    io.dump([dict, lane_id2idx], os.path.join(root_path, 'map_info.pkl'))
    for city, lane in dict.items():
        print(f'{city}: {len(lane)} lanes')

    return am


def process_dataframe(df: pd.DataFrame, am: ArgoverseMap, name, obs_len=20, obj_radius=56, lane_radius=65, valid_len=40, aug=False):
    df['TIMESTAMP'] -= np.min(df['TIMESTAMP'].values)
    seq_ts = np.unique(df['TIMESTAMP'].values)
    ts2idx = {ts: i for i, ts in enumerate(seq_ts)}

    city_name = df['CITY_NAME'].iloc[0]

    # agent
    agent_valid_index = np.array([999, -1], dtype=np.int32)
    agent_traj = np.zeros((len(seq_ts), 4), dtype=np.float32)
    for obj_type, obj_df in df.groupby('OBJECT_TYPE'):
        if obj_type == 'AGENT':
            obj_cord = obj_df[['X', 'Y']].values
            obj_ts = obj_df[['TIMESTAMP']].values.squeeze(1).tolist()
            for ts, cord in zip(obj_ts, obj_cord):
                idx = ts2idx[ts]
                agent_traj[idx, :2] = cord
                agent_traj[idx, 2] = ts
                agent_traj[idx, 3] = 1.0
                agent_valid_index[0] = min(agent_valid_index[0], idx)
                agent_valid_index[1] = max(agent_valid_index[1], idx+1)

    agent_traj_list = [agent_traj]
    agent_valid_index_list = [agent_valid_index]

    # obj
    for _, obj_df in df.groupby('TRACK_ID'):
        if obj_df['OBJECT_TYPE'].iloc[0] == 'AGENT':  # skip agent itself
            continue

        obj_valid_index = np.array([999, -1], dtype=np.int32)
        obj_traj = np.zeros((len(seq_ts), 4), dtype=np.float32)
        obj_traj[:, 2] = agent_traj[:, 2]  # timestamp
        obj_cord = obj_df[['X', 'Y']].values
        obj_ts = obj_df[['TIMESTAMP']].values.squeeze(1).tolist()
        for ts, cord in zip(obj_ts, obj_cord):
            idx = ts2idx[ts]
            obj_traj[idx, :2] = cord
            obj_traj[idx, 3] = 1.0
            obj_valid_index[0] = min(obj_valid_index[0], idx)
            obj_valid_index[1] = max(obj_valid_index[1], idx+1)
        agent_traj_list.append(obj_traj)
        agent_valid_index_list.append(obj_valid_index)

        # do pad
        start_idx = ts2idx[obj_ts[0]]   # include
        end_idx = ts2idx[obj_ts[-1]]+1  # exclude
        obj_traj[:start_idx, :2] = obj_traj[start_idx, :2]  # left
        obj_traj[end_idx:, :2] = obj_traj[end_idx-1, :2]  # right

    agent_traj_np = np.stack(agent_traj_list, axis=0)
    agent_valid_index_np = np.stack(agent_valid_index_list, axis=0)

    if aug:
        candi_indice = np.nonzero(
            agent_valid_index_np[:, 1] - agent_valid_index_np[:, 0] >= valid_len)[0]
    else:
        candi_indice = [0]

    infos = []
    for candi_idx in candi_indice:
        ed = agent_valid_index_np[candi_idx, 1]
        agent_traj = agent_traj_np.copy()

        # swap candi to first
        agent_traj[[0, candi_idx]] = agent_traj[[candi_idx, 0]]

        # chomp tail
        agent_traj = agent_traj[:, :ed]

        # pad head
        pad_size = agent_traj_np.shape[1] - agent_traj.shape[1]
        agent_traj = np.pad(
            agent_traj, ((0, 0), (pad_size, 0), (0, 0)), mode='edge')
        agent_traj[:, :pad_size, 3] = 0  # set mask to zero

        # fix timestamp, just copy the raw timestamp
        agent_traj[:, :, 2] = agent_traj_np[:, :, 2]

        # get lane
        agent_pos = agent_traj[0, obs_len, :2]
        lane_ids = am.get_lane_ids_in_xy_bbox(
            agent_pos[0], agent_pos[1], city_name, lane_radius)

        infos.append(dict(agent=agent_traj, city=city_name,
                     id=int(name)+candi_idx*10000000, lane=lane_ids))

    return infos
