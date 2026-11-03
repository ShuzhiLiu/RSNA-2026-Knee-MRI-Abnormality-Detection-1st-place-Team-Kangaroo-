"""Candidate native-neighborhood correspondence from metadata, not registration."""
import json
from pathlib import Path
import numpy as np

def frame(meta):
 assert meta['orientation_consistent'] is True,'inconsistent orientation: use independent native views'
 ori=np.asarray(meta['image_orientation_patient'],dtype=float);positions=np.asarray(meta['image_positions_patient'],dtype=float);shape=meta['shape'];spacing=np.asarray(meta['pixel_spacing'],dtype=float)
 assert ori.shape==(6,)and positions.shape==(shape[0],3)and spacing.shape==(2,)
 assert np.isfinite(ori).all()and np.isfinite(positions).all()and np.all(spacing>0)
 row,col=ori[:3],ori[3:]
 assert abs(np.linalg.norm(row)-1)<1e-3 and abs(np.linalg.norm(col)-1)<1e-3 and abs(row@col)<1e-3,'nonorthogonal metadata'
 return row,col,positions,spacing,np.cross(row,col)

def anchor(meta,index,xy):
 row,col,positions,spacing,normal=frame(meta);n,h,w=meta['shape']
 assert type(index)is int and 0<=index<n
 assert len(xy)==2 and all(np.isfinite(v)and 0<=v<=1 for v in xy)
 return positions[index]+xy[0]*(w-1)*spacing[1]*row+xy[1]*(h-1)*spacing[0]*col

def neighborhood(meta,point,diameter_mm):
 row,col,positions,spacing,normal=frame(meta);n,h,w=meta['shape'];assert np.isfinite(diameter_mm)and 20<=diameter_mm<=100
 distance=(positions-point)@normal;index=int(np.argmin(np.abs(distance)));delta=point-positions[index];x=float(delta@row/spacing[1]);y=float(delta@col/spacing[0])
 assert 0<=x<=w-1 and 0<=y<=h-1,'projected point outside this series in-plane field'
 zpos=positions@normal
 if n>1:gap=float(np.median(np.abs(np.diff(zpos))))
 else:gap=float(meta.get('slice_thickness')or 0)
 assert gap>0 and abs(distance[index])<=gap/2+1e-3,'point beyond sampled stack or irregular large gap'
 effective=max(diameter_mm,.2*w*spacing[1],.2*h*spacing[0]);width=min(float(w),effective/spacing[1]);height=min(float(h),effective/spacing[0]);left=max(0.,min(x-width/2,w-width));top=max(0.,min(y-height/2,h-height));right=left+width;bottom=top+height;indices=list(range(max(0,index-1),min(n,index+2)))
 return {'central_native_index':index,'indices':indices,'normalized_box':[left/w,top/h,right/w,bottom/h],'candidate_pixel_xy':[x,y],'nearest_plane_distance_mm':float(abs(distance[index])),'median_slice_gap_mm':gap,'requested_diameter_mm':diameter_mm,'effective_diameter_mm':float(effective),'candidate_point_in_crop':[(x-left)/width,(y-top)/height],'physical_crop_extent_mm':[(right-left)*spacing[1],(bottom-top)*spacing[0]],'metadata_correspondence_only':True,'registration_verified':False}

if __name__=='__main__':
 # Orthogonal synthetic volumes with different spacing and origins describe the same physical point.
 sag={'orientation_consistent':True,'image_orientation_patient':[0,1,0,0,0,-1],'image_positions_patient':[[i*2,0,80]for i in range(11)],'pixel_spacing':[2,1],'shape':[11,41,81]}
 ax={'orientation_consistent':True,'image_orientation_patient':[1,0,0,0,1,0],'image_positions_patient':[[0,0,i*2]for i in range(41)],'pixel_spacing':[1,1],'shape':[41,81,41]}
 point=anchor(sag,5,[.5,.5]);assert np.allclose(point,[10,40,40]);got=neighborhood(ax,point,30);assert got['central_native_index']==20 and np.allclose(got['candidate_pixel_xy'],[10,40])and got['nearest_plane_distance_mm']==0
 shifted=dict(ax,image_positions_patient=[[100,0,i*2]for i in range(41)])
 try:neighborhood(shifted,point,30)
 except AssertionError:pass
 else:raise AssertionError('out-of-field projection accepted')
 invalid=dict(ax,orientation_consistent=False)
 try:neighborhood(invalid,point,30)
 except AssertionError:pass
 else:raise AssertionError('inconsistent geometry accepted')
 print(json.dumps({'orthogonal_coordinate_fixture':True,'different_spacing_and_origin':True,'out_of_field_rejected':True,'inconsistent_geometry_rejected':True}))
