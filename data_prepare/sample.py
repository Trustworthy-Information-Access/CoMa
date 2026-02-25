import os
import pandas as pd
import random
import json
from tqdm import tqdm
random.seed(42)
image_dirs = ['./MMEB-train/N24News/train-00000-of-00001.parquet',
              './MMEB-train/HatefulMemes/train-00000-of-00001.parquet',
              './MMEB-train/VOC2007/train-00000-of-00001.parquet',
              './MMEB-train/SUN397/train-00000-of-00001.parquet',
              './MMEB-train/Visual7W/train-00000-of-00001.parquet',
              './MMEB-train/MSCOCO/train-00000-of-00001.parquet',
              './MMEB-train/VisDial/train-00000-of-00001.parquet',
              './MMEB-train/CIRR/train-00000-of-00001.parquet',
              './MMEB-train/MSCOCO_i2t/train-00000-of-00001.parquet',
              './MMEB-train/MSCOCO_t2i/train-00000-of-00001.parquet',
              './MMEB-train/NIGHTS/train-00000-of-00001.parquet',
              './MMEB-train/WebQA/train-00000-of-00001.parquet']

save_dir = '' ## sample save dir
for single in tqdm(image_dirs):
    df = pd.read_parquet(single)
    task = single.split('/')[-2]
    images = []
    for index, item in df.iterrows():
        if 'qry_image_path' in item:
            qry_image = item['qry_image_path']
            if qry_image is not None and isinstance(qry_image, str):
                images.append(qry_image)
        if 'pos_image_path' in item:
            pos_image = item['pos_image_path']
            if pos_image is not None and isinstance(pos_image, str):
                images.append(pos_image)
        if 'neg_image_path' in item:
            neg_image = item['neg_image_path']
            if neg_image is not None and isinstance(neg_image, str):
                images.append(neg_image)
    images = list(set(images))
    random.shuffle(images)
    images = images[:30000]
    images = [os.path.join('./MMEB-train', item) for item in images]
    with open(os.path.join(save_dir, task + '.json'), mode='w', encoding='utf-8') as wf:
        json.dump(images, wf, indent=4, ensure_ascii=False)

