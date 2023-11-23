# Species Classification - Machine Learning Model using RESNET-50 base

## Initial setup
1. Create a virtualenv and activate it
2. Install requirements
`pip install -r requirements.txt`

## Preparing Dataset
1. Link to download dataset
https://drive.google.com/drive/folders/1awdTI0_lXFetv8j04B87-YsLmpU3iLWT?usp=sharing
2. Save the data folder inside the project root folder(i.e inside folder ml)

NOTE: There should be three files train.csv, test.csv, vaild.csv. In case you want to add more classes, just add folders of that class with images and update these files.

## Training the model
1. Run the script
`python train.py "root_dir" n_classes n_labels`

e.g:
`python3 train.py --rot_dir=data --n_classes=8 --n_labels=2`

## Testing the model
1. Run the script
`python test.py "root_dir" n_classes n_labels`

e.g:
`python3 test.py --rot_dir=data --n_classes=8 --n_labels=2`
