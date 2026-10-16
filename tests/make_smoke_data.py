"""Generate independent synthetic fixtures. This checks plumbing, not model quality."""
from pathlib import Path
import argparse,csv
import cv2,numpy as np,yaml

def generate(root):
 root=Path(root).resolve();rng=np.random.default_rng(123)
 data=root/'source';rows=[]
 for subset,count in [('train',12),('val',4)]:
  for i in range(count):
   image=rng.integers(0,45,(128,128,3),dtype=np.uint8)
   x,y=map(int,rng.integers(28,100,2));cv2.rectangle(image,(x-12,y-12),(x+12,y+12),(40,40,235),-1)
   p=data/f'images/{subset}/{i:03d}.png';p.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(p),image)
   label=data/f'labels/{subset}/{i:03d}.txt';label.parent.mkdir(parents=True,exist_ok=True)
   label.write_text(f'0 {x/128} {y/128} {24/128} {24/128}\n')
   rows.append((p.relative_to(data).as_posix(),f'{subset}-{i}'))
 (data/'classes.txt').write_text('red_box\n')
 with (data/'groups.csv').open('w',newline='') as f:
  writer=csv.writer(f);writer.writerow(['image','group']);writer.writerows(rows)
 calib=root/'calibration';calib.mkdir(parents=True,exist_ok=True)
 for i in range(12):
  image=rng.integers(0,60,(128,128,3),dtype=np.uint8)
  cv2.rectangle(image,(20+i,30),(60+i,70),(50,50,230),-1);cv2.imwrite(str(calib/f'{i:03d}.png'),image)
 cfg={'project':{'name':'smoke'},'classes':{'mode':'custom_only','custom_names':['red_box']},
      'data':{'source_dir':str(data),'split':{'mode':'provided','expected_counts':{'train':12,'val':4,'test':0}},
              'calibration_source':'directory','calibration_dir':str(calib)},
      'model':{'pretrained':None,'input_hw':[64,96]},
      'train':{'epochs':1,'batch':2,'workers':0,'device':'cpu','amp':False,'mosaic':0.,'close_mosaic':0},
      'quantize':{'calibration':{'max_samples':12,'device':'cpu'},'precision_level':0,'finetune_level':0}}
 p=root/'config.yaml';p.write_text(yaml.safe_dump(cfg));print(p)
 return p
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('directory');args=parser.parse_args();generate(args.directory)
