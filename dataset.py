import torchvision.datasets as datasets

class ImageFolderWithName(datasets.ImageFolder):
    def __init__(self, return_fn=False, *args, **kwargs):
        super(datasets.ImageFolder, self).__init__(*args, **kwargs)
        self.return_fn = return_fn

    def __getitem__(self, i):
        img, label = super(ImageFolderWithName, self).__getitem__(i)
        if not self.return_fn:
            return img, label
        else:
            return img, label, self.imgs[i]