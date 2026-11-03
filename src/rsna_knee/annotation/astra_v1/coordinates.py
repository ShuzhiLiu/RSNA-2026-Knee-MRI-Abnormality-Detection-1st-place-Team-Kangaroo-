"""Source-cell image-edge coordinate contract, separate from native rendering."""
import math
import numpy as np

def manifest(series, indices, crop, window=None, pad=0, tile=640):
    n,h,w=series['shape'];x0,y0,x1,y1=crop
    left,top=max(0,math.floor(x0*w)),max(0,math.floor(y0*h))
    right,bottom=min(w,math.ceil(x1*w)),min(h,math.ceil(y1*h))
    spacing=series['pixel_spacing'];paired=window is not None
    cols=2 if paired or len(indices)>1 else 1
    rows=len(indices) if paired else math.ceil(len(indices)/cols)
    pw,ph=(right-left)*spacing[1],(bottom-top)*spacing[0]
    scale=min(tile/pw,tile/ph);nw,nh=max(1,round(pw*scale)),max(1,round(ph*scale))
    cells=[]
    for j,idx in enumerate(indices):
        for k in range(2 if paired else 1):
            xx,yy=(k*tile,j*(tile+36)) if paired else ((j%cols)*tile,(j//cols)*(tile+36))
            ox,oy=xx+(tile-nw)//2+pad,yy+36+(tile-nh)//2+pad
            cells.append({'series_key':series['key'],'series_uid':series['series_uid'],
                'native_index':idx,'channel':'adjusted' if k else 'original',
                'source_wh':[w,h],'crop_xyxy':[left,top,right,bottom],
                'image_rect_xyxy':[ox,oy,ox+nw,oy+nh]})
    return {'sheet_wh':[cols*tile+2*pad,rows*(tile+36)+2*pad],
        'coordinate_definition':'xyxy continuous normalized image edges relative to entire VIEW sheet, x right y down; must be inside exactly one image_rect excluding titles and padding','cells':cells}

def to_native(box, meta):
    W,H=meta['sheet_wh'];xy=np.array(box['xyxy'],float)*[W,H,W,H]
    assert np.isfinite(xy).all() and 0<=xy[0]<xy[2]<=W and 0<=xy[1]<xy[3]<=H,'invalid xyxy'
    cells=[c for c in meta['cells'] if c['series_key']==box['series_key'] and c['native_index']==box['native_index'] and c['channel']==box['channel']]
    assert len(cells)==1,'wrong series/slice/channel'
    c=cells[0];l,t,r,b=c['image_rect_xyxy'];a,d,e,f=c['crop_xyxy']
    assert xy[0]>=l-1e-8 and xy[1]>=t-1e-8 and xy[2]<=r+1e-8 and xy[3]<=b+1e-8,'box crosses image-cell boundary'
    return ((xy-[l,t,l,t])*[(e-a)/(r-l),(f-d)/(b-t),(e-a)/(r-l),(f-d)/(b-t)]+[a,d,a,d]).tolist()

def to_sheet(native, cell, wh):
    l,t,r,b=cell['image_rect_xyxy'];a,d,e,f=cell['crop_xyxy']
    return (((np.array(native)-[a,d,a,d])*[(r-l)/(e-a),(b-t)/(f-d),(r-l)/(e-a),(b-t)/(f-d)]+[l,t,l,t])/np.array([*wh,*wh])).tolist()

def iou(a,b):
    x=max(0,min(a[2],b[2])-max(a[0],b[0]));y=max(0,min(a[3],b[3])-max(a[1],b[1]))
    inter=x*y;union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union>0 else 0.
