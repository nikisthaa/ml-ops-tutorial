import argparse
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, confusion_matrix

from model import MultiTaskModel
from dataset import Custom_BaMI_Dataset


def test_model(model, dataloader, device, labels, classes, output_dir):
  model.eval() # Set model to evaluation mode
  correct_classes = 0 # Species
  correct_labels = 0  # Poi/No-Poi
  total = 0
  all_class_targets = []
  all_label_targets = []
  all_class_preds = []
  all_label_preds = []
  with torch.no_grad():
    for i, data in enumerate(dataloader):
      # get the image and send device
      inputs = data['image'].to(device)
      labels = data['labels'].to(device)
      classes = data['classes'].to(device)

      # Forward passs
      classes_outputs, labels_outputs = model(inputs)

      # Get predictions
      _, class_predicted = torch.max(classes_outputs.data, 1)
      _, label_predicted = torch.max(labels_outputs.data, 1)

      # Extend our lists for sklearns metrics functions
      all_class_targets.extend(classes.cpu().numpy())
      all_label_targets.extend(labels.cpu().numpy())
      all_class_preds.extend(class_predicted.cpu().numpy())
      all_label_preds.extend(label_predicted.cpu().numpy())

      # Update the correct and total counts
      total += labels.size(0)
      correct_classes += (classes == class_predicted).sum().item()

      correct_labels += (labels == label_predicted).sum().item()

  # Calculate confusion matrices
  class_confusion_matrix = confusion_matrix(all_class_targets, all_class_preds)
  label_confusion_matrix = confusion_matrix(all_label_targets, all_label_preds)

  # Calculate precision and recall for classes
  class_precision = precision_score(all_class_targets, all_class_preds, average='macro')
  class_recall = recall_score(all_class_targets, all_class_preds, average='macro')

  # Calculate precision and recall for labels
  label_precision = precision_score(all_label_targets, all_label_preds)
  label_recall = recall_score(all_label_targets, all_label_preds)

  print('For Classes - Precision: {:.2f}% Recall: {:.2f}%'.format(class_precision * 100, class_recall * 100))
  print('For Labels - Precision: {:.2f}% Recall: {:.2f}%'.format(label_precision * 100, label_recall * 100))

  print('Accuracy of the network on test images: Classes: {:.2f}% Labels: {:.2f}%'.format(100 * correct_classes/total, 100 * correct_labels/total))

  print(type(class_confusion_matrix))
  # Create a pandas dataframe for pretty printing our confusion matrices
  class_df = pd.DataFrame(class_confusion_matrix, index=classes_y.values(), columns=classes_y.values())
  label_df = pd.DataFrame(label_confusion_matrix, index=labels_y.values(), columns=labels_y.values())

  plt.figure(figsize=(10, 7))
  sns.heatmap(class_df, annot=True)
  plt.title('Confusion matrix for classes')
  plt.ylabel('True class')
  plt.xlabel('Predicted class')
  plt.savefig(f'{output_dir}/class_confusion_matrix.png')

  plt.figure(figsize=(10, 7))
  sns.heatmap(label_df, annot=True)
  plt.title('Confusion matrix for labels')
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.savefig(f'{output_dir}/label_confusion_matrix.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training the model.")
    parser.add_argument('--input_size', type=int, default=256, help="The input size for image")
    parser.add_argument('--batch_size', type=int, default=32, help="The batch size for training")
    parser.add_argument('--output_dir', type=str, default="output", help="The output directory to store the result")
    parser.add_argument('--shuffle', type=bool, default=True,help="The boolean value to shuffle/not shuffle the data")
    parser.add_argument('--num_workers', type=str, default=1, help="The number of workers to load dataset for training")
    parser.add_argument('--n_classes', type=int, help="The number of classes to classify (i.e number of species)")
    parser.add_argument('--n_labels', type=int, help="The number of labels to classify (i.e for edibility)")
    parser.add_argument('--root_dir', type=str, help="The root directory where data is located")

    args = parser.parse_args()

    INPUT_SIZE = args.input_size
    ROOT_DIR = args.root_dir
    OUTPUT_DIR = args.output_dir
    BATCH_SIZE = args.batch_size
    SHUFFLE = args.shuffle
    NUM_WORKERS = args.num_workers
    N_CLASSES = args.n_classes
    N_LABELS = args.n_labels

    # NOTE: You can add more classes and labels as per the dataset
    # Create a dictionary that maps key to values
    labels_y = {
        '0': 'Non-Poisonous',
        '1': 'Poisonous'
    }

    classes_y = {
        '0': 'Angel',
        '1': 'Death',
        '2': 'Elder',
        '3': 'Misletoe',
        '4': 'Cherrys',
        '5': 'CloudBerrys',
        '6': 'Lion',
        '7': 'Oyster'
    }

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ]
    )

    # Load test dataset
    test_dataset = Custom_BaMI_Dataset(csv_file = 'test.csv', root_dir = ROOT_DIR, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define our model
    model = MultiTaskModel(N_CLASSES, N_LABELS)

    # 3. Load weights
    model.load_state_dict(torch.load(f'{OUTPUT_DIR}/best_model.pt', map_location=device))

    model = model.to(device)
    test_model(model, test_dataloader, device, labels_y, classes_y, OUTPUT_DIR)
