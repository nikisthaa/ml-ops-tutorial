import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class MultiTaskModel(nn.Module):
  # n_classes : number of species - 8
  # n_labels : poisonos/non-poisonous
  def __init__(self, n_classes, n_labels):
    super(MultiTaskModel, self).__init__()

    # initialize base model ( RestNet )
    self.base_model = models.resnet50(pretrained=True)
    num_features = self.base_model.fc.in_features

    # Freeze the base model
    for param in self.base_model.parameters():
        param.requires_grad = False

    # Removes the last layer
    self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])

    # Define new layers
    # Create a dense layer with input feat = num_features, and output features = 128
    self.fc = nn.Linear(num_features, 128)
    self.fc_classes = nn.Linear(128, n_classes)
    self.fc_labels = nn.Linear(128, n_labels)

  def forward(self, x):
    x = self.base_model(x)
    x = x.view(x.size(0), -1) # Flatten the tensors
    x = F.relu(self.fc(x))

    species_output = self.fc_classes(x)
    poisonous_output = self.fc_labels(x)

    return species_output, poisonous_output
