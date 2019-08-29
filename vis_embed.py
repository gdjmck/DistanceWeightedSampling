import torch
from tensorboardX import SummaryWriter
import pickle
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

if __name__ == '__main__':
	writer = SummaryWriter()
	with open('embedding_toy25_epoch44.pkl', 'rb') as f:
		data = pickle.load(f)
	labels = [key.split('/')[-2] for key in data.keys()]
	feats = list(data.values())
	imgs = list(data.keys())
	num = 8000
	labels = labels[:num]
	imgs = imgs[:num]
	feats = feats[:num]
	to_tensor = transforms.Compose([transforms.Resize(50), transforms.RandomCrop((50, 50)), transforms.ToTensor()])
	imgs = [to_tensor(Image.open(img).convert('RGB')).unsqueeze(0) for img in imgs]
	
	imgs = torch.cat(imgs, 0)	
	feats = [torch.Tensor(feat[0]) for feat in feats]
	feats = torch.cat(feats, 0)
	writer.add_embedding(feats, metadata=labels, label_img=imgs)
	writer.close()
