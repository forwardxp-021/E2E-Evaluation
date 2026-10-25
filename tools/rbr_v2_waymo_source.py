#!/usr/bin/env python3
"""Minimal, typed read-only Motion protobuf projection (unknown fields retained).

Field numbers/types follow hash-bound official scenario.proto/map.proto. This is
not a substitute full Waymo SDK and never invents a segment ID or global time.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
import struct


def scenario_class():
    f = descriptor_pb2.FileDescriptorProto(name='rbr_motion_projection.proto', package='rbr_motion', syntax='proto2')
    def msg(name, fields):
        m = f.message_type.add(name=name)
        for name, number, typ, repeated in fields:
            x = m.field.add(name=name, number=number, label=3 if repeated else 1)
            if isinstance(typ, str):
                x.type = 11; x.type_name = '.rbr_motion.'+typ
            else:
                x.type = typ
    # protobuf FieldDescriptorProto: double1 float2 int64 3 int32 5 bool8 string9 bytes12.
    msg('ObjectState', [(k, n, t, False) for k, n, t in [
        ('center_x',2,1),('center_y',3,1),('center_z',4,1),('length',5,2),('width',6,2),
        ('height',7,2),('heading',8,2),('velocity_x',9,2),('velocity_y',10,2),('valid',11,8)]])
    msg('Track', [('id',1,5,False),('object_type',2,5,False),('states',3,'ObjectState',True)])
    msg('Point', [('x',1,1,False),('y',2,1,False),('z',3,1,False)])
    msg('Neighbor', [(k,n,t,False) for k,n,t in [('feature_id',1,3),('self_start_index',2,5),
        ('self_end_index',3,5),('neighbor_start_index',4,5),('neighbor_end_index',5,5)]])
    msg('Lane', [('type',2,5,False),('polyline',8,'Point',True),('entry_lanes',9,3,True),
                 ('exit_lanes',10,3,True),('left_neighbors',11,'Neighbor',True),('right_neighbors',12,'Neighbor',True)])
    msg('MapFeature', [('id',1,3,False),('lane',3,'Lane',False)])
    msg('Scenario', [('timestamps_seconds',1,1,True),('tracks',2,'Track',True),
        ('scenario_id',5,9,False),('sdc_track_index',6,5,False),('map_features',8,'MapFeature',True),
        ('current_time_index',10,5,False),('compressed_frame_laser_data',12,12,True),
        ('frame_camera_tokens',13,12,True)])
    p = descriptor_pool.DescriptorPool(); p.Add(f)
    return message_factory.MessageFactory(p).GetPrototype(p.FindMessageTypeByName('rbr_motion.Scenario'))


Scenario = scenario_class()


def records(path):
    """Yield offsets/payloads; whole TFRecord SHA is verified by build, CRC not claimed."""
    with open(path, 'rb') as f:
        while True:
            offset = f.tell(); head = f.read(12)
            if not head:
                break
            if len(head) != 12:
                raise ValueError('Truncated TFRecord header')
            size = struct.unpack('<Q', head[:8])[0]
            if size > 64*1024*1024:
                raise ValueError('Oversized TFRecord')
            payload = f.read(size); tail = f.read(4)
            if len(payload) != size or len(tail) != 4:
                raise ValueError('Truncated TFRecord payload/CRC')
            yield offset, payload, head, tail


def load_record(path, offset, size):
    with open(path, 'rb') as f:
        f.seek(int(offset)+12); payload = f.read(int(size))
    if len(payload) != int(size):
        raise ValueError('Missing raw record bytes')
    s = Scenario(); s.ParseFromString(payload)
    return s
