import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image, ImageFile
import os

default_path = "/home/data/frames20220628/"  # 文件夹目录


class MyDataset(Dataset):
    def __init__(self, path=default_path, transform=None):
        '''
            read img from path and do the transformation
            save img name as a list
        '''
        self.name_list = []
        self.data = []
        self.transform = transform
        for filename in os.listdir(path):
            if filename.endswith(".jpeg"):
                self.name_list.append(filename)
                img = self.read_image((path+filename))
                label = int(filename[:3])
                camid = int(filename[4])
                self.data.append((img, camid, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        '''
            return img, camid, label, filename
        '''
        img, camid, label = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, camid, label, self.name_list[index]

    def read_image(self, img_path):
        """Keep reading image until succeed.
        This can avoid IOError incurred by heavy IO process."""
        got_img = False
        if not os.path.exists(img_path):
            raise IOError("{} does not exist".format(img_path))
        while not got_img:
            try:
                img = Image.open(img_path).convert('RGB')
                got_img = True
            except IOError:
                print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(
                    img_path))
        return img


if __name__ == '__main__':
    data_transforms = transforms.Compose([
        transforms.Resize((256, 256), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    mydataset = MyDataset(default_path, data_transforms)
    dataloader = DataLoader(mydataset, batch_size=16)
    for imgs, camids, labels, pic_names in dataloader:
        print(f'imgs.shape: {imgs.shape} \
              \nlabels: {labels} \
              \ncamids:{camids} \
              \npic_names: {pic_names}')
    with torch.no_grad():
        pass
        # model = model.eval()
