import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
def overview(q):
    v=np.load(q['volume_path'],mmap_mode='r');assert v.dtype==np.uint16
    tile=384;cols=6;rows=math.ceil(len(q['indices'])/cols)
    sheet=Image.new('RGB',(cols*tile,rows*(tile+22)),'black');draw=ImageDraw.Draw(sheet);font=ImageFont.load_default(size=15)
    for j,idx in enumerate(q['indices']):
        arr=np.rint(v[idx].astype(np.float32)/257).astype(np.uint8);h,w=arr.shape;sp=q['pixel_spacing']or[1,1]
        scale=min(tile/(w*sp[1]),tile/(h*sp[0]));nw,nh=max(1,round(w*sp[1]*scale)),max(1,round(h*sp[0]*scale))
        im=Image.fromarray(arr).convert('RGB').resize((nw,nh),Image.Resampling.LANCZOS);x,y=j%cols*tile,j//cols*(tile+22)
        sheet.paste(im,(x+(tile-nw)//2,y+22+(tile-nh)//2));draw.text((x+3,y+2),f'{q["key"]} {idx}',fill='white',font=font)
    return sheet
