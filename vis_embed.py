import torch
from tensorboardX import SummaryWriter
import pickle
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='Visualize embedding in tensorboard.')
	parser.add_argument('--embeddings', type=str, required=True, help='embedding pickle file.')
	parser.add_argument('--save-folder', type=str, default='./runs', help='directory to save tensorboard file.')
	return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()
	assert args.embeddings.endswith('.pkl')
	writer = SummaryWriter(args.save_folder)
	with open(args.embeddings, 'rb') as f:
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
