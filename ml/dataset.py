import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class Custom_BaMI_Dataset(Dataset):
  def __init__(self, csv_file, root_dir, transform=None):
    self.labels_frame = pd.read_csv(os.path.join(root_dir, csv_file))
    self.root_dir = root_dir
    self.transform = transform

    self.type_to_int = {type: i for i, type in enumerate(self.labels_frame.iloc[:, 2].unique())}

  def __len__(self): #To show the total number of data to represent in our file
    return len(self.labels_frame)

  def __getitem__(self, idx): #To get infividual image
    if torch.is_tensor(idx):
      idx = idx.toList() #Convert tensor to a list

    img_name = os.path.join(self.root_dir, self.labels_frame.iloc[idx, 0])
    image = Image.open(img_name)
    if self.transform:
      image = self.transform(image)

    labels = torch.tensor(1 if self.labels_frame.iloc[idx, 1] == 'yes' else 0, dtype = torch.long)
    classes = torch.tensor(self.type_to_int[self.labels_frame.iloc[idx, 2]], dtype = torch.long)


    data = {'image' : image, 'labels' : labels, 'classes' : classes}
    return data
