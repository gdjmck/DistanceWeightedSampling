import torch
import torchvision
import pickle
from PIL import Image
import train
import os

def loader_test(fn):
    parts = fn.split('/')
    label = parts[-2]
    return fn, label, Image.open(fn).convert('RGB')

def dict_reverse(d):
    d_rev = {}
    for key in d.keys():
        d_rev[d[key]] = key
    return d_rev

if __name__ == '__main__':
    device = torch.device('cuda')
    args = train.args
    model = train.model
    model.eval()

    data = train.convert_dataset(args.data_path)
    fn_dict = dict_reverse(data.class_to_idx)
    # print('class to idx:', data.class_to_idx)
    # print(len(data.targets), data.targets)

    dataset = torch.utils.data.DataLoader(data)
    embeddings = {}
    with torch.no_grad():
        for i, (img, label) in enumerate(dataset):
            label = fn_dict[label.numpy()[0]]
            print(label)
            img = img.to(device)
            embedding = model(img, sampling=False).cpu().numpy()
            if label not in embeddings.keys():
                embeddings[label] = [embedding]
            else:
                embeddings[label].append(embedding)
        
    with open('embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
        print('saved embedding.')        