import os
import csv
import torch
import argparse
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss

from model import MultiTaskModel
from utils import calculate_accuracy
from dataset import Custom_BaMI_Dataset


def train(model, dataloader, optimizer, criterion_label, criterion_class, device):
  epoch_loss = 0
  epoch_acc = 0

  model.train()
  for i, data in enumerate(dataloader):
    # Get images, labels, class from data and move to device
    inputs = data['image'].to(device)
    labels = data['labels'].to(device)
    classses = data['classes'].to(device)

    # Initialize our optmizers
    optimizer.zero_grad()

    # Forward pass
    classes_outputs, labels_outputs = model(inputs)

    # Calculate loss
    loss_classes = criterion_class(classes_outputs, classses)
    loss_labels = criterion_label(labels_outputs, labels)

    # Calculate acc
    acc_classes = calculate_accuracy(classes_outputs, classses)
    acc_labels = calculate_accuracy(labels_outputs, labels)

    loss = loss_classes + loss_labels
    acc = (acc_classes + acc_labels)/2

    # Backward propagation
    loss.backward()
    optimizer.step()

    epoch_loss += loss.item()
    epoch_acc += acc.item()

  return epoch_loss/len(dataloader), epoch_acc/len(dataloader)


def evaluate(model, dataloader, criterion_label, criterion_class, device):
  epoch_loss = 0
  epoch_acc = 0

  model.eval()
  with torch.no_grad():
    for i, data in enumerate(dataloader):
      # Get images, labels, class from data and move to device
      inputs = data['image'].to(device)
      labels = data['labels'].to(device)
      classses = data['classes'].to(device)

      # Forward pass
      classes_outputs, labels_outputs = model(inputs)

      # Calculate loss
      loss_classes = criterion_class(classes_outputs, classses)
      loss_labels = criterion_label(labels_outputs, labels)

      # Calculate acc
      acc_classes = calculate_accuracy(classes_outputs, classses)
      acc_labels = calculate_accuracy(labels_outputs, labels)

      loss = loss_classes + loss_labels
      acc = (acc_classes + acc_labels)/2

      epoch_loss += loss.item()
      epoch_acc += acc.item()

  return epoch_loss/len(dataloader), epoch_acc/len(dataloader)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training the model.")
    parser.add_argument('--input_size', type=int, default=256, help="The input size for image")
    parser.add_argument('--batch_size', type=int, default=32, help="The batch size for training")
    parser.add_argument('--output_dir', type=str, default="output", help="The output directory to store the result")
    parser.add_argument('--shuffle', type=bool, default=True,help="The boolean value to shuffle/not shuffle the data")
    parser.add_argument('--num_workers', type=str, default=1, help="The number of workers to load dataset for training")
    parser.add_argument('--epoch', type=int, default=20, help="The number of epochs for training")
    parser.add_argument('--lr', type=float, default=0.01, help="The learning rate for training")
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
    EPOCH = args.epoch
    LEARNING_RATE = args.lr
    N_CLASSES = args.n_classes
    N_LABELS = args.n_labels

    print(f"Training the model with epoch: {EPOCH}, batch size: {BATCH_SIZE}, learning rate: {LEARNING_RATE}")
    # Check if the output directory exists
    if not os.path.exists(OUTPUT_DIR):
        # If it doesn't exist, create it
        os.makedirs(OUTPUT_DIR)

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ]
    )

    train_dataset = Custom_BaMI_Dataset(csv_file = 'train.csv', root_dir = ROOT_DIR, transform=transform)
    valid_dataset = Custom_BaMI_Dataset(csv_file = 'vaild.csv', root_dir = ROOT_DIR, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define our model
    model = MultiTaskModel(N_CLASSES, N_LABELS).to(device)

    # Define the loss function and optimizers
    loss_fn_classes = CrossEntropyLoss()
    loss_fn_labels = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    best_valid_loss = float('inf')
    # Open the file in append mode to store out loss, acc
    with open(f'{OUTPUT_DIR}/metrics.csv', 'w') as file:
        writer = csv.writer(file)
        # write the headers
        writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy'])

        # Training LOOP
        for epoch in range(EPOCH):
            train_loss, train_acc = train(model, train_dataloader, optimizer, loss_fn_labels, loss_fn_classes, device)
            valid_loss, valid_acc = evaluate(model, valid_dataloader, loss_fn_labels, loss_fn_classes, device)

            # write the metrics to the CSV file
            writer.writerow([epoch, train_loss, train_acc, valid_loss, valid_acc])

            print(f'Epoch: {epoch+1:02}')
            print(f'\t Train Loss: {train_loss: .3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Valied Loss: {valid_loss: .3f} | Valid Acc: {valid_acc*100:.2f}%')

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), f'{OUTPUT_DIR}/best_model.pt') # Save the model

    # save the last model
    torch.save(model.state_dict(), f'{OUTPUT_DIR}/last_model.pt')
