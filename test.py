import torch
import train
import os
import glob
import numpy as np
import visdom
import cv2
from PIL import Image
import torchvision.transforms as transforms

model = train.model

transform = transforms.Compose([
        transforms.Resize((228)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        train.normalize
])


if __name__ == '__main__':
    model.eval()
    args = train.args
    data = glob.glob(os.path.join(train.traindir, '*.jpg'))
    print('num of data:', len(data))
    np.random.shuffle(data)
    ref = data[:5]
    vis = visdom.Visdom()
    for f in ref:
        vis.image(np.transpose(cv2.imread(f)[..., ::-1], (2, 0, 1)), win=f)    

    with torch.no_grad():
        rec = {f.rsplit('/')[-1]: {'top_8': [], 'query': model(transform(Image.open(f).convert('rgb'))).detach().numpy()} for f in ref}

        for f in data[5:]:
            img = Image.open(f).convert('rgb')
            inp = transform(img)
            embedding = model(inp).detach().numpy()
            for key in rec.keys():
                rec[key]['top_8'].append({'fn': f, 'dist': np.fabs(embedding - rec[key]['query']).sum()})
                if len(rec[key]['top_8']) > 8:
                    last_fn = rec[key]['top_8'][-1]['fn']
                    rec[key]['top_8'] = sorted(rec[key]['top_8'], key=lambda x: x['dist'])[:8]
                    update = False
                    for d in rec[key]['top_8']:
                        if d['fn'] == last_fn:
                            update = True
                            break
                    if update:
                        # update top 8 in visdom
                        imgs = np.concatenate([np.transpose(cv2.resize(cv2.imread(os.path.join(args.img_folder_test, d['fn'])), (250, 250))[..., ::-1], (2, 0, 1))[np.newaxis] for d in top_4[j]['top_8']])
                        vis.images(imgs, win=j, nrow=2, opts=dict(title='IMG_%s'%key))
                        print('Updated %s'%key)