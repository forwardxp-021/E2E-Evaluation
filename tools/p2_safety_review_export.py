#!/usr/bin/env python3
"""Export a figure and event timeline from immutable read-only diagnostics."""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.p2_safety_readonly_review import OUT,ROOT,SCIENCE,TARGET,read,sha,write_new


def export():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Polygon as Patch
    from matplotlib.lines import Line2D
    from shapely.geometry import shape
    from shapely.ops import unary_union
    from shapely.affinity import translate
    import numpy as np
    font=Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    if not font.exists():raise FileNotFoundError('Chinese plot font absent')
    matplotlib.font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':FontProperties(fname=str(font)).get_name(),'axes.unicode_minus':False,'svg.hashsalt':'P2_SAFETY_V1','svg.fonttype':'path'})
    d=read(OUT/'evidence/P2_Safety_Readonly_Diagnostics_v1.json');f=d['frames'];expert=d['expert_reference_frames'];ev=d['collision_events'][0]
    origin=(589060.,4474620.)
    xy=lambda x,y:np.column_stack([np.array(x)-origin[0],np.array(y)-origin[1]])
    actual=xy([x['center_x'] for x in f],[x['center_y'] for x in f]);ex=xy([x['center_x'] for x in expert],[x['center_y'] for x in expert])
    da=unary_union([shape(x['geometry']) for x in d['map_geometry']['drivable_polygons']])
    fig,axes=plt.subplots(1,3,figsize=(18,7),layout='constrained',gridspec_kw={'width_ratios':[1.2,1,1]})
    def patches(ax,geometry,fc,ec,lw=.5,alpha=1,zorder=0):
        g=translate(geometry,xoff=-origin[0],yoff=-origin[1])
        for p in ([g] if g.geom_type=='Polygon' else list(g.geoms)):
            if p.geom_type!='Polygon':continue
            ax.add_patch(Patch(np.array(p.exterior.coords),facecolor=fc,edgecolor=ec,lw=lw,alpha=alpha,zorder=zorder))
            for hole in p.interiors:ax.add_patch(Patch(np.array(hole.coords),facecolor='#f5f2ed',edgecolor=ec,lw=lw,zorder=zorder+.01))
    for ax in axes:
        ax.set_facecolor('#f5f2ed');patches(ax,da,'#e4eee9','#4d7365',.7)
        for lane in d['map_geometry']['lanes']:
            line=shape(lane['baseline']);coords=np.array(line.coords);ax.plot(coords[:,0]-origin[0],coords[:,1]-origin[1],color='#a8b5ae',lw=.5,ls=':',zorder=1)
        ax.plot(ex[:,0],ex[:,1],color='#3a78a4',lw=1.8,ls='--',zorder=3)
        ax.plot(actual[:,0],actual[:,1],color='#b96738',lw=2,zorder=4)
        ax.plot(actual[42:70,0],actual[42:70,1],color='#cb3e43',lw=3,zorder=5)
        ax.set_aspect('equal');ax.grid(alpha=.15);ax.set_xlabel('地图东向偏移（m）');ax.set_ylabel('地图北向偏移（m）')
        ax.tick_params(labelsize=9)
    def footprint(ax,i,color):
        from shapely.geometry import Polygon
        coords=[(c['x'],c['y']) for c in f[i]['corners']]
        patches(ax,Polygon(coords),'none',color,1.6,zorder=7)
    a=axes[0];a.set_title('A 既有真实轨迹与道路结构',fontsize=12)
    for i,color,lab in [(0,'#333333','起点'),(44,'#ba343d','4.4秒 越界超容差'),(74,'#7c2e87','7.4秒 碰撞')]:
        a.scatter(*actual[i],s=35,color=color,zorder=9)
        offset=(8,-5) if i==44 else (8,7)
        a.annotate(lab,actual[i],xytext=offset,textcoords='offset points',fontsize=9,color=color)
    a.set_xlim(-5,47);a.set_ylim(-5,38)
    a.legend(handles=[Line2D([0],[0],color='#b96738',lw=2,label='已运行Baseline车体中心'),
        Line2D([0],[0],color='#3a78a4',ls='--',label='原始expert（描述参考）'),
        Line2D([0],[0],color='#cb3e43',lw=3,label='至少一个角点越界的已采样段')],loc='lower left',fontsize=8,framealpha=.95)
    b=axes[1];b.set_title('B 4.4秒：右前角距可行驶区域0.376m',fontsize=12)
    footprint(b,44,'#b3343f')
    c=f[44]['corners'][3];q=c['nearest_drivable_xy'];cx=c['x']-origin[0];cy=c['y']-origin[1]
    b.plot([cx,q[0]-origin[0]],[cy,q[1]-origin[1]],color='#b3343f',lw=2,zorder=9)
    b.scatter(cx,cy,color='#b3343f',s=30,zorder=9)
    b.annotate('右前角FR\n0.375552m > 0.30m',xy=(cx,cy),xytext=(15,22),textcoords='offset points',fontsize=10,arrowprops={'arrowstyle':'->','color':'#b3343f'},color='#b3343f')
    footprint(b,51,'#96456a');b.text(.03,.03,'紫色轮廓：5.1秒最大角点越界0.677m\n中心仍在可行驶区域内',transform=b.transAxes,fontsize=9)
    b.set_xlim(14,31);b.set_ylim(21,32)
    cax=axes[2];cax.set_title('C 7.4秒：与静止目标车辆首次盒重叠',fontsize=12)
    patches(cax,shape(ev['object_box']),'#d7cae5','#684086',1.5,alpha=.9,zorder=6)
    patches(cax,shape(ev['ego_box']),'none','#b96738',1.7,zorder=7)
    overlap=shape(ev['object_box']).intersection(shape(ev['ego_box']));patches(cax,overlap,'#cc3347','#cc3347',1,zorder=8)
    p=ev['overlap_centroid'];p0=(p[0]-origin[0],p[1]-origin[1]);cax.scatter(*p0,s=28,color='#cc3347',zorder=10)
    cax.annotate('重叠区质心\n面积0.012827m²',xy=p0,xytext=(5,-45),textcoords='offset points',fontsize=9,color='#a72a3d',arrowprops={'arrowstyle':'->','color':'#a72a3d'})
    center=(ev['object_center'][0]-origin[0],ev['object_center'][1]-origin[1]);cax.annotate('目标车：日志速度0\ntrack 7e7c1fcccb0b58ff',center,xytext=(-8,35),textcoords='offset points',fontsize=9,ha='center',color='#684086')
    cax.text(.03,.03,'自车4.990m/s；纵向请求0\n此刻全部车体角点已回到可行驶区域',transform=cax.transAxes,fontsize=9)
    cax.set_xlim(-1,17);cax.set_ylim(22,35)
    fig.suptitle('P2第7对Baseline安全诊断：越界先发生，碰撞在另一地点随后发生',fontsize=15)
    fig.supxlabel('坐标原点：(589060, 4474620)；未修改地图/车体轮廓。事件由冻结记录按官方定义只读重建，未运行新仿真。',fontsize=10)
    for suffix in ['png','svg']:
        path=OUT/('P2_Safety_Trajectory_And_Events_v1.'+suffix)
        if path.exists():raise FileExistsError(path)
        fig.savefig(path,dpi=160,bbox_inches='tight',metadata={'Date':None} if suffix=='svg' else None)
    plt.close(fig)
    svg=OUT/'P2_Safety_Trajectory_And_Events_v1.svg';svg.write_text('\n'.join(x.rstrip() for x in svg.read_text().splitlines())+'\n')
    events=[]
    for i,event,evidence,comment in [(0,'INITIAL_STATE','FROZEN_CURRENT_EGO_AND_LOG','初态无碰撞、角点合规；原始初加速度为负'),
        (21,'BASELINE_BRAKE_REQUEST_START','FROZEN_FINAL_COMMAND','从此帧发出−1.45；当前model状态尚响应上一请求'),
        (39,'BASELINE_BRAKE_REQUEST_END','FROZEN_FINAL_COMMAND','发出0；当前model减速度仍有滤波尾巴'),
        (42,'FIRST_CORNER_OUTSIDE','OFFICIAL_CORNERS_PARQUET_TIME_SERIES_PLUS_READ_ONLY_CORNER_LOCALIZATION','官方序列首次0；FR越界0.114257m，未超0.3m'),
        (44,'FIRST_CORNER_BEYOND_0P3M','READ_ONLY_RECONSTRUCTION_FROM_OFFICIAL_GEOMETRY_DEFINITION','FR越界0.375552m；全程compliance失败条件首次成立'),
        (51,'MAXIMUM_CORNER_DISTANCE','READ_ONLY_GEOMETRY_DIAGNOSTIC','FR最大越界0.677095m'),
        (69,'FIRST_ALL_CORNERS_INSIDE_AFTER_EXCURSION','OFFICIAL_CORNERS_PARQUET_TIME_SERIES','4.2～6.8秒共27个越界采样帧结束；全程compliance不会恢复'),
        (73,'LAST_SAMPLED_NONCOLLISION','READ_ONLY_OFFICIAL_COLLISION_GEOMETRY','与目标盒距离0.099225m；只用于约束首次采样重叠区间'),
        (74,'FIRST_COLLISION_WITH_STOPPED_TRACK','READ_ONLY_RECONSTRUCTION_USING_OFFICIAL_COLLISION_HELPERS','STOPPED_TRACK_COLLISION；不伪称parquet提供事件时间/ID'),
        (79,'LAST_RECORDED_STATE','FROZEN_CURRENT_EGO_AND_LOG','仍有目标盒重叠；无下一次控制推进')]:
        row=f[i];corner=max(row['corners'],key=lambda x:x['nearest_drivable_distance_m'] or 0)
        events.append(dict(frame_index_zero_based=i,display_frame_one_based=i+1,episode_time_s=row['relative_s'],simulation_timestamp_us=row['time_us'],
            raw_log_timestamp_us=row['raw_log_time_us'],raw_minus_sim_us=row['raw_minus_sim_us'],event=event,evidence_type=evidence,
            ego_center_x=row['center_x'],ego_center_y=row['center_y'],ego_rear_x=row['rear_x'],ego_rear_y=row['rear_y'],
            ego_speed_mps=row['speed_mps'],current_model_acceleration_mps2=row['model_accel_mps2'],
            outgoing_longitudinal_request_mps2=row['outgoing_request_mps2'] if i<79 else '',
            current_outside_corner=corner['corner'] if row['outside'] else '',current_max_corner_distance_m=corner['nearest_drivable_distance_m'] if row['outside'] else 0.,
            collided_track_token=ev['track_token'] if i==74 else '',
            diagnostic_overlap_centroid_x=ev['overlap_centroid'][0] if i==74 else '',diagnostic_overlap_centroid_y=ev['overlap_centroid'][1] if i==74 else '',comment=comment))
    with (OUT/'P2_Safety_Key_Event_Timeline_v1.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(events[0]),lineterminator='\n');writer.writeheader();writer.writerows(events)
    write_new(OUT/'evidence/P2_Safety_Figure_And_Timeline_Provenance_v1.json',dict(
        diagnostics_sha256=sha(OUT/'evidence/P2_Safety_Readonly_Diagnostics_v1.json'),export_source_sha256=sha(Path(__file__)),
        figure_origin=list(origin),polygon_edits=False,geometry_buffer_or_simplification=False,intermediate_continuous_collision_time='UNKNOWN',
        first_sampled_overlap_interval_s=[7.3,7.4],overlap_centroid_is_physical_contact_point=False,scientific_simulation=0,
        output_sha256={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'P2_Safety_Trajectory_And_Events_v1.png',svg,OUT/'P2_Safety_Key_Event_Timeline_v1.csv']}))


if __name__=='__main__':export()
