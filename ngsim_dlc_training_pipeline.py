#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NGSIM US-101 自动数据处理：轨迹 -> 主线换道代理事件 -> 六车交互 -> 三分类训练样本。
Python >= 3.9; numpy, pandas; pyarrow 强烈推荐（无 pyarrow 自动用 csv.gz）。

约束与定义：
  * 只有三份轨迹，不能知道驾驶员真实动机，1-5 主线相邻变道仅作 DLC proxy。
  * 正类=未来 2~5 秒内发生一个可靠的主线相邻车道 Lane_ID 转换，
    预测时刻不出现明显横向运动，且为该轨迹遇到的下一个原始车道变化。
  * 负类=接下来 5 秒轨迹连续、没有任何 Lane_ID 变化。
  * 其余情况（临近转换、轨迹截尾/间断、车道 ID 抖动等）不标注，不作为负类。
  * 训练特征严格使用预测时刻及以前数据；不会输出未来标注为输入特征。

生成：
  lane_change_events.csv                候选事件+原因标记
  dataset_details/<时段>.parquet/csv.gz    训练样本+邻车原始字段+标识（不能整体喂模型）
  model_data/train|val|test.parquet/csv.gz  仅 feature_* 与 label, 可直接用于训练
  quality_report.json                  自动质量检查
  feature_columns.json                 确切训练特征名单

运行（默认路径已在 CONFIG 中写好）：
    python ngsim_dlc_training_pipeline.py
    python ngsim_dlc_training_pipeline.py --smoke-test
    python ngsim_dlc_training_pipeline.py --outdir "D:\\anaxin\\NGSIM\\my_output"
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


FT_TO_M = 0.3048
NAMES18 = [
    'Vehicle_ID','Frame_ID','Total_Frames','Global_Time','Local_X','Local_Y',
    'Global_X','Global_Y','v_Length','v_Width','v_Class','v_Vel','v_Acc',
    'Lane_ID','Preceding','Following','Space_Headway','Time_Headway',
]
NEED = [
    'Vehicle_ID','Frame_ID','Global_Time','Local_X','Local_Y',
    'v_Length','v_Width','v_Class','v_Vel','v_Acc',
    'Lane_ID','Preceding','Following',
]

@dataclass
class Config:
    inputs: Tuple[str, ...] = (
        r'D:\anaxin\NGSIM\trajectories_0750am_0805am.txt',
        r'D:\anaxin\NGSIM\trajectories_0805am_0820am.txt',
        r'D:\anaxin\NGSIM\trajectories_0820am_0835am.txt',
    )
    segments: Tuple[str, ...] = ('0750_0805','0805_0820','0820_0835')
    outdir: str = r'D:\anaxin\NGSIM\dlc_processed'
    main_lanes: Tuple[int, ...] = (1,2,3,4,5)
    ego_class: int = 2  # NGSIM v_Class: passenger auto=2; 邻车不限类型
    fps: int = 10
    sample_every_frames: int = 5   # 每 0.5 秒产生候选训练时刻
    future_horizon_s: float = 5.0
    positive_min_lead_s: float = 2.0
    stable_before_s: float = 0.8
    stable_after_s: float = 0.8
    min_history_s: float = 1.0
    after_any_change_guard_s: float = 2.0
    max_abs_lateral_speed_mps: float = 0.65
    max_abs_lane_center_offset_m: float = 0.75
    ewm_span: int = 5
    lane_geom_bin_m: float = 25.0
    # 超过物理允许的一帧位移则断开轨迹，不把两段轨迹连起来构造标签
    max_one_step_longitudinal_m: float = 15.0
    max_one_step_lateral_m: float = 5.0


DTYPES = {
    'Vehicle_ID':'float64', 'Frame_ID':'float64', 'Global_Time':'float64',
    'Local_X':'float32','Local_Y':'float32',
    'v_Length':'float32','v_Width':'float32','v_Class':'float32',
    'v_Vel':'float32','v_Acc':'float32',
    'Lane_ID':'float32','Preceding':'float32','Following':'float32',
}


def read_ngsim(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'原始文件不存在：{path}\n请核对文件路径与扩展名。')
    # 只探测第一行，不重复加载数百万条完整数据
    with path.open('r', encoding='utf-8-sig', errors='replace') as f:
        first = f.readline().strip()
    if not first:
        raise ValueError(f'空文件：{path}')
    sep = ',' if ',' in first else r'\s+'
    parts = re.split(r',|\s+', first)
    normalize = lambda x: re.sub(r'[^a-z0-9]','',x.lower())
    is_header = normalize(parts[0]) == 'vehicleid'
    opts = dict(sep=sep, low_memory=False, encoding='utf-8-sig', on_bad_lines='error')
    if is_header:
        raw_cols = pd.read_csv(path, nrows=0, **opts).columns.tolist()
        ref = {normalize(c):c for c in raw_cols}
        missing = [c for c in NEED if normalize(c) not in ref]
        if missing:
            raise ValueError(f'{path.name} 表头缺少字段：{missing}')
        actual = [ref[normalize(c)] for c in NEED]
        df = pd.read_csv(path, usecols=actual, dtype={ref[normalize(c)]:DTYPES[c] for c in NEED}, **opts)
        df = df.rename(columns={ref[normalize(c)]:c for c in NEED})
    else:
        if len(parts) != 18:
            raise ValueError(f'{path.name} 首行有 {len(parts)} 列，预期 NGSIM 官方 18 列。请核对文件。')
        df = pd.read_csv(path, header=None, names=NAMES18, usecols=NEED, dtype=DTYPES, **opts)
    for c in NEED:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def clean_and_derive(raw: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    n0 = len(raw)
    required = ['Vehicle_ID','Frame_ID','Global_Time','Local_X','Local_Y',
                'v_Length','v_Width','v_Class','v_Vel','Lane_ID']
    raw = raw.dropna(subset=required).copy()
    raw = raw.loc[(raw['v_Length']>1) & (raw['v_Length']<100)
                  & (raw['v_Width']>0) & (raw['v_Width']<25)
                  & (raw['v_Vel']>=-5) & (raw['v_Vel']<200)
                  & (raw['Lane_ID'].between(1,8))].copy()
    for c in ['Vehicle_ID','Frame_ID','Global_Time']:
        raw[c] = raw[c].astype('int64')
    for c in ['v_Class','Lane_ID','Preceding','Following']:
        raw[c] = raw[c].fillna(0).astype('int32')
    raw = raw.sort_values(['Vehicle_ID','Frame_ID','Global_Time'], kind='mergesort')
    dupe = int(raw.duplicated(['Vehicle_ID','Frame_ID'], keep='first').sum())
    raw = raw.drop_duplicates(['Vehicle_ID','Frame_ID'], keep='first').reset_index(drop=True)

    # 只有真实逐帧相邻的记录才构成连续轨迹，避免按车辆ID拼接丢失区间
    prev = raw.groupby('Vehicle_ID', sort=False)
    fd = prev['Frame_ID'].diff()
    td = prev['Global_Time'].diff()
    yd = prev['Local_Y'].diff().abs() * FT_TO_M
    xd = prev['Local_X'].diff().abs() * FT_TO_M
    new_track = (fd.ne(1) | td.lt(70) | td.gt(130)
                 | yd.gt(cfg.max_one_step_longitudinal_m)
                 | xd.gt(cfg.max_one_step_lateral_m))
    raw['track_id'] = new_track.cumsum().astype('int32')
    raw['x_m'] = (raw['Local_X']*FT_TO_M).astype('float32')
    raw['y_m'] = (raw['Local_Y']*FT_TO_M).astype('float32')
    raw['length_m'] = (raw['v_Length']*FT_TO_M).astype('float32')
    raw['width_m'] = (raw['v_Width']*FT_TO_M).astype('float32')
    raw['speed_mps'] = (raw['v_Vel']*FT_TO_M).astype('float32')
    raw['acc_mps2'] = (raw['v_Acc']*FT_TO_M).astype('float32')
    raw['time_s'] = ((raw['Global_Time']-int(raw['Global_Time'].min()))/1000).astype('float32')
    raw['x_smooth_m'] = raw.groupby('track_id', sort=False)['x_m'].transform(
        lambda s: s.ewm(span=cfg.ewm_span, adjust=False).mean()
    ).astype('float32')
    raw['a_smooth_mps2'] = raw.groupby('track_id',sort=False)['acc_mps2'].transform(
        lambda s: s.ewm(span=cfg.ewm_span, adjust=False).mean()
    ).astype('float32')
    # 因果（单边）横向速度，不用中心差分/未来帧
    dxx = raw.groupby('track_id',sort=False)['x_smooth_m'].diff()
    dtt = raw.groupby('track_id',sort=False)['Global_Time'].diff()/1000.0
    raw['v_lat_mps'] = (dxx/dtt).astype('float32')

    metrics = {
        'raw_rows':int(n0), 'valid_rows':int(len(raw)),
        'invalid_rows_dropped':int(n0-len(raw)),
        'duplicate_vehicle_frame_dropped':dupe,
        'unique_vehicle_ids':int(raw['Vehicle_ID'].nunique()),
        'continuous_tracks':int(raw['track_id'].nunique()),
        'ego_rows':int((raw['v_Class']==cfg.ego_class).sum()),
    }
    return raw,metrics


def lane_centers(raw: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, Dict[int, Tuple[np.ndarray,np.ndarray]]]:
    # 按路段纵坐标分箱估计车道中心线；用于当前横向偏移和换道开始前保守筛选
    d = raw.loc[raw['Lane_ID'].isin(cfg.main_lanes), ['Lane_ID','x_m','y_m']].copy()
    d['y_bin'] = np.floor(d['y_m']/cfg.lane_geom_bin_m).astype('int32')
    ctr = d.groupby(['Lane_ID','y_bin'], observed=True)['x_m'].agg(['median','count']).reset_index()
    ctr = ctr.rename(columns={'median':'center_x_m', 'count':'n'})
    ctr = ctr.loc[ctr['n']>=10].copy()
    centers = {}
    for lane,g in ctr.groupby('Lane_ID'):
        g=g.sort_values('y_bin')
        centers[int(lane)] = (g['y_bin'].to_numpy(dtype=np.float64)*cfg.lane_geom_bin_m+cfg.lane_geom_bin_m/2,
                              g['center_x_m'].to_numpy(dtype=np.float64))
    del d
    # 只对中心线可估计的车道赋值，避免缺省假设误判
    raw['lane_center_m'] = np.nan
    for lane,(ys,xs) in centers.items():
        mask = raw['Lane_ID'].eq(lane)
        if mask.any():
            ys_query = raw.loc[mask,'y_m'].to_numpy(dtype=np.float64)
            raw.loc[mask,'lane_center_m'] = np.interp(ys_query,ys,xs).astype('float32')
    raw['lane_offset_m'] = (raw['x_smooth_m']-raw['lane_center_m']).astype('float32')
    return ctr,centers


def extract_events(raw:pd.DataFrame,cfg:Config) -> Tuple[pd.DataFrame,Dict[Tuple[int,int],int]]:
    min_before=int(round(cfg.stable_before_s*cfg.fps))
    min_after=int(round(cfg.stable_after_s*cfg.fps))
    events=[]
    event_lookup={}
    # 对连续 track 分析：Lane_ID连续恒定段，前后足够稳定才确认转换
    for tid,g in raw.groupby('track_id',sort=False):
        lane=g['Lane_ID'].to_numpy()
        frame=g['Frame_ID'].to_numpy()
        t=g['Global_Time'].to_numpy()
        boundaries=np.r_[0,np.flatnonzero(lane[1:]!=lane[:-1])+1,len(g)]
        for a,b,c in zip(boundaries[:-2],boundaries[1:-1],boundaries[2:]):
            s=int(lane[a]); d=int(lane[b]); n1=b-a;n2=c-b
            stable = n1>=min_before and n2>=min_after
            main_adj = s in cfg.main_lanes and d in cfg.main_lanes and abs(s-d)==1
            car = int(g['v_Class'].iloc[b])==cfg.ego_class
            valid = bool(stable and main_adj and car)
            info={
                'segment':'', 'vehicle_id':int(g['Vehicle_ID'].iloc[0]),
                'track_id':int(tid), 'frame_cross':int(frame[b]),
                'global_time_cross_ms':int(t[b]), 'source_lane':s,'target_lane':d,
                'direction':('LCL' if d<s else 'LCR' if d>s else 'UNKNOWN'),
                'pre_stable_frames':int(n1),'post_stable_frames':int(n2),
                'is_mainline_adjacent':bool(main_adj), 'stable_both_sides':bool(stable),
                'ego_is_car':bool(car), 'valid_proxy_event':valid,
                'remark':'mainline_DLC_proxy_not_true_intent'
            }
            events.append(info)
            if valid:
                event_lookup[(int(tid),int(frame[b]))] = (1 if d<s else 2)
    cols=['segment','vehicle_id','track_id','frame_cross','global_time_cross_ms',
          'source_lane','target_lane','direction','pre_stable_frames',
          'post_stable_frames','is_mainline_adjacent','stable_both_sides',
          'ego_is_car','valid_proxy_event','remark']
    return pd.DataFrame(events,columns=cols),event_lookup


def choose_training_moments(raw:pd.DataFrame, events_lookup:Dict[Tuple[int,int],int],
                            cfg:Config) -> Tuple[pd.DataFrame,Dict[str,int]]:
    horizon=int(round(cfg.future_horizon_s*cfg.fps))
    min_lead=int(round(cfg.positive_min_lead_s*cfg.fps))
    min_history=int(round(cfg.min_history_s*cfg.fps))
    guard=int(round(cfg.after_any_change_guard_s*cfg.fps))
    idxs=[]; labs=[]; lead=[]
    report={'candidate_downsampled':0,'complete_past_future':0,'calm_predecision':0,
            'negative':0,'positive_left':0,'positive_right':0,
            'unknown_dropped':0}
    for tid,g in raw.groupby('track_id',sort=False):
        if len(g)<min_history+horizon+1:
            continue
        f=g['Frame_ID'].to_numpy(); lane=g['Lane_ID'].to_numpy()
        cl=g['v_Class'].to_numpy(); lat=g['v_lat_mps'].to_numpy()
        offset=g['lane_offset_m'].to_numpy()
        validcar=(cl==cfg.ego_class)&np.isin(lane,cfg.main_lanes)
        candidates=np.flatnonzero(validcar & (f%cfg.sample_every_frames==0))
        report['candidate_downsampled']+=len(candidates)
        candidates=candidates[(candidates>=min_history)&(candidates+horizon<len(g))]
        report['complete_past_future']+=len(candidates)
        # 当前阶段保守限制横向操作迹象，以免预测已经在执行的换道
        calm = (np.isfinite(lat[candidates])
                & (np.abs(lat[candidates])<=cfg.max_abs_lateral_speed_mps)
                & np.isfinite(offset[candidates])
                & (np.abs(offset[candidates])<=cfg.max_abs_lane_center_offset_m))
        candidates=candidates[calm]
        report['calm_predecision']+=len(candidates)
        if not len(candidates):
            continue
        changes=np.flatnonzero(lane[1:]!=lane[:-1])+1
        nxtpos=np.searchsorted(changes,candidates,side='right')
        prevpos=nxtpos-1
        distance_prev=np.where(prevpos>=0,
            candidates - changes[np.clip(prevpos,0,max(len(changes)-1,0))] if len(changes) else 99999,
            99999)
        usable=distance_prev>=guard
        for j,p,ok in zip(candidates,nxtpos,usable):
            if not ok:
                report['unknown_dropped']+=1
                continue
            if p>=len(changes) or changes[p]>j+horizon:
                label=0; ttc=np.nan
            else:
                crossing=changes[p]
                if not (min_lead<=crossing-j<=horizon):
                    report['unknown_dropped']+=1
                    continue
                label=events_lookup.get((int(tid),int(f[crossing])))
                if label is None:
                    report['unknown_dropped']+=1
                    continue
                ttc=(int(f[crossing])-int(f[j]))/cfg.fps
            idxs.append(int(g.index[j]));labs.append(label);lead.append(ttc)
            report[('negative' if label==0 else 'positive_left' if label==1 else 'positive_right')]+=1
    if idxs:
        samples=raw.loc[idxs].copy()
        samples['label']=np.asarray(labs,dtype='int8')
        samples['time_to_cross_s_AUDIT_ONLY']=np.asarray(lead,dtype='float32')
    else:
        samples=raw.iloc[:0].copy()
        samples['label']=pd.Series(dtype='int8')
        samples['time_to_cross_s_AUDIT_ONLY']=pd.Series(dtype='float32')
    samples=samples.sort_values(['Frame_ID','Vehicle_ID']).reset_index(drop=True)
    samples['sample_id']=np.arange(len(samples),dtype='int32')
    return samples,report


ROLES={'cf':(0,'forward'),'cr':(0,'backward'),
       'lf':(-1,'forward'),'lr':(-1,'backward'),
       'rf':(1,'forward'),'rr':(1,'backward')}


def neighbor_interactions(raw:pd.DataFrame,samples:pd.DataFrame,cfg:Config)->pd.DataFrame:
    if samples.empty:
        return pd.DataFrame()
    ego=samples.rename(columns={
        'Vehicle_ID':'ego_id','Lane_ID':'ego_lane','x_m':'ego_x_m','y_m':'ego_y_m',
        'length_m':'ego_length_m','width_m':'ego_width_m',
        'v_Class':'ego_class','speed_mps':'ego_speed_mps',
        'a_smooth_mps2':'ego_acc_mps2', 'v_lat_mps':'ego_v_lat_mps',
        'lane_offset_m':'ego_lane_offset_m',
    }).copy()
    # 只留下被采样时帧的全部车辆：为降内存，仍包括卡车/摩托车作为邻车
    selected_frames=samples['Frame_ID'].drop_duplicates()
    cand=raw.loc[raw['Frame_ID'].isin(selected_frames),[
        'Frame_ID','Lane_ID','Vehicle_ID','x_m','y_m',
        'speed_mps','a_smooth_mps2','length_m','width_m','v_Class',
    ]].copy()
    cand=cand.rename(columns={'Vehicle_ID':'nei_id','Lane_ID':'search_lane',
         'x_m':'nei_x_m','y_m':'nei_y_m','speed_mps':'nei_speed_mps',
         'a_smooth_mps2':'nei_acc_mps2','length_m':'nei_length_m',
         'width_m':'nei_width_m','v_Class':'nei_class'})
    cand=cand.sort_values('nei_y_m',kind='mergesort').reset_index(drop=True)
    for role,(delta,direction) in ROLES.items():
        # 注意 asof 要求先按纵向位置整体排序，不可仅按 frame/lane 排序
        q=ego[['sample_id','Frame_ID','ego_lane','ego_y_m']].copy()
        q['search_lane']=(q['ego_lane']+delta).astype('int32')
        q=q.loc[q['search_lane'].isin(cfg.main_lanes)]
        if q.empty:
            for name in ['id','lane','x_m','y_m','speed_mps','acc_mps2','length_m',
                         'width_m','class','rel_y_m','rel_v_mps','clearance_m','ttc_s','exists']:
                ego[f'{role}_{name}']=np.nan
            continue
        q=q.sort_values('ego_y_m',kind='mergesort')
        merged=pd.merge_asof(
            q,cand,left_on='ego_y_m',right_on='nei_y_m',
            by=['Frame_ID','search_lane'],direction=direction,
            allow_exact_matches=False
        )
        # 位置完全相等（罕见）时为了绝对避免自车匹配，不启用 exact_matches
        merged=merged.set_index('sample_id')
        payload={
            'id':'nei_id','lane':'search_lane','x_m':'nei_x_m','y_m':'nei_y_m',
            'speed_mps':'nei_speed_mps','acc_mps2':'nei_acc_mps2',
            'length_m':'nei_length_m','width_m':'nei_width_m','class':'nei_class',
        }
        for target,origin in payload.items():
            vals=merged[origin].reindex(ego['sample_id']).to_numpy()
            ego[f'{role}_{target}']=vals
        ego[f'{role}_exists']=ego[f'{role}_id'].notna().astype('int8')
        ego[f'{role}_lane']=ego[f'{role}_lane'].where(ego[f'{role}_exists']==1)
        ego[f'{role}_rel_y_m']=ego[f'{role}_y_m']-ego['ego_y_m']
        ego[f'{role}_rel_v_mps']=ego[f'{role}_speed_mps']-ego['ego_speed_mps']
        if direction=='forward':
            ego[f'{role}_clearance_m']=(ego[f'{role}_y_m']-ego[f'{role}_length_m']-ego['ego_y_m'])
            closing=ego['ego_speed_mps']-ego[f'{role}_speed_mps']
        else:
            ego[f'{role}_clearance_m']=(ego['ego_y_m']-ego['ego_length_m']-ego[f'{role}_y_m'])
            closing=ego[f'{role}_speed_mps']-ego['ego_speed_mps']
        gap=ego[f'{role}_clearance_m']
        ego[f'{role}_ttc_s']=np.where((gap>0)&(closing>0.1),gap/np.maximum(closing,0.1),np.nan)
        del merged,q
    return ego


def make_training_features(detail:pd.DataFrame,cfg:Config)->Tuple[pd.DataFrame,List[str]]:
    result=pd.DataFrame(index=detail.index)
    def add(name,values):
        result['feature_'+name]=pd.to_numeric(values,errors='coerce').astype('float32')
    add('ego_lane',detail['ego_lane'])
    add('ego_speed_mps',detail['ego_speed_mps'])
    add('ego_acc_mps2',detail['ego_acc_mps2'])
    add('ego_v_lat_mps',detail['ego_v_lat_mps'])
    add('ego_lane_offset_m',detail['ego_lane_offset_m'])
    add('ego_length_m',detail['ego_length_m'])
    add('ego_width_m',detail['ego_width_m'])
    add('left_mainline_exists',(detail['ego_lane']>min(cfg.main_lanes)).astype('int8'))
    add('right_mainline_exists',(detail['ego_lane']<max(cfg.main_lanes)).astype('int8'))
    for role in ROLES:
        for x in ['exists','speed_mps','acc_mps2','length_m','width_m','class',
                  'rel_y_m','rel_v_mps','clearance_m','ttc_s']:
            add(role+'_'+x,detail[role+'_'+x])
    result['label']=detail['label'].to_numpy(dtype='int8')
    return result,[x for x in result.columns if x.startswith('feature_')]


def parquet_ok()->bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def write_data(df:pd.DataFrame,path:Path,use_parquet:bool)->str:
    path.parent.mkdir(parents=True,exist_ok=True)
    if use_parquet:
        file=path.with_suffix('.parquet')
        df.to_parquet(file,index=False,engine='pyarrow',compression='zstd')
    else:
        file=path.with_suffix('.csv.gz')
        df.to_csv(file,index=False,compression='gzip')
    return str(file)


def json_dump(data:Any,path:Path):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str),encoding='utf-8')


def run(cfg:Config):
    out=Path(cfg.outdir)
    out.mkdir(parents=True,exist_ok=True)
    use_parquet=parquet_ok()
    if not use_parquet:
        print('[提示] 未安装 pyarrow，输出将使用 csv.gz；建议 pip install pyarrow')
    for p in cfg.inputs:
        if not Path(p).is_file():
            raise FileNotFoundError(f'文件不存在：{p}')
    reports=[];events_all=[];feature_columns=None
    for segment,path in zip(cfg.segments,cfg.inputs):
        print(f'\n===== {segment} =====',flush=True)
        print('[1/6] 读取数据……',flush=True)
        raw=read_ngsim(Path(path))
        print('[2/6] 清洗与单侧平滑……',flush=True)
        state,report=clean_and_derive(raw,cfg)
        del raw;gc.collect()
        print('[3/6] 车道中心线与换道事件……',flush=True)
        centers,funcs=lane_centers(state,cfg)
        events,lookup=extract_events(state,cfg)
        events['segment']=segment
        events_all.append(events)
        report['mainline_lane_centers']=sorted(funcs)
        report['all_lane_transitions']=len(events)
        report['proxy_events']=int(events['valid_proxy_event'].sum())
        report['proxy_LCL']=int((events['valid_proxy_event']&(events['direction']=='LCL')).sum())
        report['proxy_LCR']=int((events['valid_proxy_event']&(events['direction']=='LCR')).sum())
        print('[4/6] 生成无未来泄漏的时刻标签……',flush=True)
        samples,label_report=choose_training_moments(state,lookup,cfg)
        report.update(label_report)
        print('[5/6] 匹配当前车道及左右车道六类邻车……',flush=True)
        detail=neighbor_interactions(state,samples,cfg)
        del state,samples,centers,funcs,lookup;gc.collect()
        if detail.empty:
            raise RuntimeError(f'{segment} 无可用样本，请检查输入字段、轨迹连续性与阈值。')
        # 对照原始 NGSIM Preceding/Following（这些字段不进入训练特征）
        for role,col in [('cf','Preceding'),('cr','Following')]:
            subset=(detail[col]>0)&detail[f'{role}_id'].notna()
            report[f'{role}_raw_ID_compared']=int(subset.sum())
            report[f'{role}_raw_ID_agreement']=(float((detail.loc[subset,col].to_numpy()==
                    detail.loc[subset,f'{role}_id'].to_numpy()).mean()) if subset.any() else None)
        for role in ROLES:
            report[f'{role}_neighbor_found_rate']=float(detail[f'{role}_exists'].mean())
            ok=detail[f'{role}_clearance_m'].notna()
            report[f'{role}_negative_clearance_rate']=(float((detail.loc[ok,f'{role}_clearance_m']<0).mean())
                 if ok.any() else None)
        print('[6/6] 保存训练和审计文件……',flush=True)
        report['segment']=segment
        write_data(detail,out/'dataset_details'/segment,use_parquet)
        model,feature_columns=make_training_features(detail,cfg)
        split={'0750_0805':'train','0805_0820':'val','0820_0835':'test'}[segment]
        write_data(model,out/'model_data'/split,use_parquet)
        reports.append(report)
        print(f"{segment}: 轨迹行数={report['valid_rows']:,}, 有效代理换道={report['proxy_events']:,}, "
              f"模型样本={len(model):,}（0:{report['negative']:,} / 1:{report['positive_left']:,} / 2:{report['positive_right']:,}）",
              flush=True)
        del detail,model,events;gc.collect()
    pd.concat(events_all,ignore_index=True).to_csv(out/'lane_change_events.csv',index=False,encoding='utf-8-sig')
    json_dump(asdict(cfg),out/'processing_config.json')
    json_dump({'segments':reports,
              'label_definition':{'0':'5s 内无任何 Lane_ID 变化','1':'未来2~5s 主线向左变道代理事件',
                                  '2':'未来2~5s 主线向右变道代理事件'},
              'limitations':['不能由轨迹证明驾驶员真实非强制换道意图',
                             '跨线前安静筛选仅降低换道进行中标签污染，无法保证全部是换道开始前',
                             '仅基于 1-5 主线相邻车道稳定变化作为代理标签',
                             '预测不同时间段性能存在交通状况分布变化']},out/'quality_report.json')
    json_dump(feature_columns,out/'feature_columns.json')
    if any(r['positive_left']+r['positive_right']==0 for r in reports):
        print('[警告] 有时段没有正类，请查看 quality_report.json。')
    print('\n处理结束。输出目录：',out,flush=True)
    print('训练输入只用 model_data 中 feature_* 列；label 是预测目标。')


def build_smoke_data()->pd.DataFrame:
    """构造 120 帧共 7 车：1号左变、2号右变，含6类邻车。"""
    records=[]
    # lane 2: left front/rear; lane 3: current front/rear; lane 4: right front/rear
    specs=[(1,3,100,2),(2,3,130,2),(3,3,70,2),
           (4,2,130,3),(5,2,70,2),(6,4,130,2),(7,4,70,2)]
    # 车道越靠左 x 越小，纵向方向 y 越大
    for f in range(120):
        for vid,lane,y0,vclass in specs:
            nowlane = (2 if f>=80 else lane) if vid==1 else ((4 if f>=95 else lane) if vid==2 else lane)
            x = nowlane*12.0
            if vid==1 and 68<=f<80:
                x=3*12-(f-68)/12.0*12
            if vid==2 and 83<=f<95:
                x=3*12+(f-83)/12.0*12
            y=y0+f*0.5
            row=[vid,f,120,1_000_000+100*f,x,y,0.0,0.0,15,6,vclass,16.4042,0,nowlane,0,0,0,0]
            records.append(row)
    return pd.DataFrame(records,columns=NAMES18)[NEED]


def smoke_test(cfg:Config):
    r=build_smoke_data()
    state,diag=clean_and_derive(r,cfg)
    ctr,fn=lane_centers(state,cfg)
    ev,lk=extract_events(state,cfg)
    assert int(ev['valid_proxy_event'].sum())==2,ev.to_string(index=False)
    s,meta=choose_training_moments(state,lk,cfg)
    assert (s['label']==1).sum()>0,meta
    assert (s['label']==2).sum()>0,meta
    detail=neighbor_interactions(state,s,cfg)
    assert not detail.empty
    pre=detail[(detail['ego_id']==1)&(detail['Frame_ID']<70)].copy()
    assert not pre.empty
    test=pre.iloc[0]
    expected={'cf':2,'cr':3,'lf':4,'lr':5,'rf':6,'rr':7}
    for role,id_expected in expected.items():
        assert int(test[f'{role}_id'])==id_expected, (role,test[f'{role}_id'],id_expected)
    model,features=make_training_features(detail,cfg)
    assert not any('time_to_cross' in c for c in features)
    assert not any(c.endswith('_id') or c.endswith('_track_id') for c in features)
    assert set(model['label'].unique()).issubset({0,1,2})
    print('SMOKE TEST PASSED: event, 6 neighbors, classes, leakage exclusion.')
    print('labels=',model['label'].value_counts().sort_index().to_dict(),
          '| feature_columns=',len(features))


def main():
    parser=argparse.ArgumentParser(description='NGSIM US-101 主线 DLC 代理标签 + 6 车交互 + 三分类数据生成')
    parser.add_argument('--outdir',default=None)
    parser.add_argument('--inputs',nargs=3,default=None)
    parser.add_argument('--smoke-test',action='store_true')
    args=parser.parse_args()
    cfg=Config()
    if args.outdir:cfg.outdir=args.outdir
    if args.inputs:cfg.inputs=tuple(args.inputs)
    if args.smoke_test:smoke_test(cfg)
    else:run(cfg)

if __name__=='__main__':
    main()
