"""Generate ARTIFICIAL metal-like images for functional demonstrations only."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFilter

def generate(root='data/demo',count=80,seed=42):
    rng=np.random.default_rng(seed)
    for label in ('defective','non_defective'):
        folder=Path(root)/label;folder.mkdir(parents=True,exist_ok=True)
        for i in range(count):
            # Synthetic brushed-metal appearance, with lighting/noise variation.
            base=rng.integers(95,175); rows=rng.normal(0,8,(128,1,1)); noise=rng.normal(0,10,(128,128,1))
            arr=np.clip(base+rows+noise+np.array([2,0,-2]),0,255).astype('uint8')
            im=Image.fromarray(arr,'RGB'); d=ImageDraw.Draw(im)
            if label=='defective':
                for j in range(int(rng.integers(2,6))):
                    x=int(rng.integers(10,110));y=int(rng.integers(10,110));length=int(rng.integers(15,65))
                    points=[(x,y),(min(127,x+length//2),max(0,min(127,y+int(rng.integers(-12,12))))),(min(127,x+length),max(0,min(127,y+int(rng.integers(-15,15)))))]
                    d.line(points,fill=(int(rng.integers(10,50)),)*3,width=int(rng.integers(2,5)))
            im.save(folder/f'{label}_{i:03}.png')
    print('Created synthetic demo images in',root)
if __name__=='__main__':generate()
