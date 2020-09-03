import glob
import os
# from shutil import copyfile
from pprint import pprint
import pickle
from PIL import Image
import numpy as np
from autoPyTorch import AutoNetImageClassification, AutoNetClassification, HyperparameterSearchSpaceUpdates
import sklearn.model_selection
import sklearn.metrics
from sklearn import preprocessing
import torch
from torch.autograd import Variable
from torchvision import transforms
import json
import pandas as pd
from pathlib import Path, PurePath
import shutil

import argparse

parser = argparse.ArgumentParser(description='Chexpert automl and hyperparameter search.')
parser.add_argument('--input_size', nargs='+', type=int, default=[8, 8])
parser.add_argument('--channels', type=int, default=3)
parser.add_argument('--balance', type=int, default=280, help='Examples / class, set None if you want to load all examples')
parser.add_argument('--work_location', type=str, default='/tmp/chexpert_dataset')
parser.add_argument('--chexpertpath', type=str, default='/nfs/managed_datasets/chexpert/CheXpert-v1.0-small')
parser.add_argument('--preset', type=str, default='full_cs')
parser.add_argument('--save_output_to', type=str, default='/tmp')
parser.add_argument('--prepare_data', type=bool, default=False)
args = parser.parse_args()
input_size = tuple(args.input_size)
channels = args.channels
balance = args.balance
work_location = args.work_location
chexpertpath = args.chexpertpath
preset = args.preset
save_output_to = "{3}/{1}balanced_ba_ce_{0}_{2}".format(input_size, balance, preset, args.save_output_to)


def get_chexpert_metadata(path):
    datacsv = os.path.join(path)
    df = pd.read_csv(datacsv)
    df = df.fillna(0)
    df.describe().to_csv("chexpert_description_{}.csv".format("_".format(path.split(os.pathsep))))
    labels = df.columns.values[5:]
    return labels, df


def write_chexpert_metadata(df, path, phase):
    counter = 0
    for i, row in df.iterrows():
        if row[3] == "Frontal": # Loading only frontal
            positive = np.where(row[5:].values == 1.0)[0] 
            if len(positive) == 1: # Loading only categorical examples (max 1 pathology)
                with open("{}/{}_{}.txt".format("indexes", phase, "_".join(df.columns.values[5:][positive[0]].split(" "))), "a", newline='\n') as fp:
                    counter += 1
                    fp.write(os.path.join(path, *row['Path'].split("/")[1:]))
                    # fp.write('\n')
    print("Wrote {} examples in total".format(counter)) #Default settings run yields


def prepare_dataset(chexpertpath = "/nfs/managed_datasets/chexpert/CheXpert-v1.0-small"):
    print("Chexpert reader")
    train_labels, train_df = get_chexpert_metadata(os.path.join(chexpertpath, "train.csv"))
    valid_labels, valid_df = get_chexpert_metadata(os.path.join(chexpertpath, "valid.csv"))
    assert set(valid_labels)==set(train_labels), "We assume training and validation sets exactly the same labels"
    print("Loading {1} pathologies from Chexpert: {0}".format(train_labels, len(train_labels))) #Default settings run yields 14 pathologies
    write_chexpert_metadata(train_df, chexpertpath, "train")
    write_chexpert_metadata(valid_df, chexpertpath, "valid")


def get_data_references(images = [] , labels = [] , filter = "train_", copy_to = work_location, metadata_location="indexes", balance = balance, img_channels = channels, copy=False):
    '''
    If you don't  want to copy set copy_to = None; only if this is set img_channels is taken into account 
    balance = 180 means each class gets 180 examples before splits
    '''
    for name in glob.glob(os.path.join(metadata_location, "*{}*".format(filter))):
        pname = Path(name).with_suffix('').parts[-1].lower() #select the metainfofilename. We expect the filenames are formed out of a concatenated phase and pathologyname. eg, train_Pneumothorax.txt, valid_Lung_Opacity.txt, etc.
        class_name = " ".join(pname.split(filter)[-1].split("_"))
        for i, imagepaths in enumerate(open(name).readlines()):
            if i==balance:
                break
            images.append(imagepaths)
            labels.append(class_name)
    le = preprocessing.LabelBinarizer()
    le.fit(list(set(labels)))
    #Create sklearn type labels
    labels = le.transform(labels)
    print("Found {} examples with labels {}".format(len(labels), le.classes_))
    assert len(labels) > 0, "No data found"
    #Copy data locally and preprocess
    if copy_to:
        os.makedirs(copy_to, exist_ok=True)
        for i, im in enumerate(images):
            dest = Path(os.path.join(copy_to, str(i))).with_suffix(Path(im).suffix.strip())
            if copy:
                if "\n" in im:
                    im = im.strip()
                if img_channels == 3:
                    mode = "RGB"
                elif img_channels == 1:
                    mode = "L"
                with Image.open(im).convert(mode) as image:
                    image = image.resize(input_size)
                    im_arr = np.fromstring(image.tobytes(), dtype=np.uint8) / 255.0
                try:
                    im_arr = im_arr.reshape((image.size[1], image.size[0], img_channels))
                except ValueError as e:
                    im_arr = im_arr.reshape((image.size[1], image.size[0]))
                    im_arr = np.stack((im_arr,) * img_channels, axis=-1)
                finally:
                    im_arr = np.moveaxis(im_arr, -1, 0) #Pytorch is channelfirst
                    # copyfile(im, dest)
                    image.save(dest) 
                    # print("Wrote image {} of shape {} with label {}({})".format(dest, im_arr.shape, labels[i], le.inverse_transform(labels)[i]))
            images[i] = dest
        print("Found {} samples".format(i+1))
    assert len(images) == len(labels)
    return images, labels, le


def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.cuda()  #assumes that you're using GPU


def jpg_image_to_array(image_path):
    """
    Loads JPEG imag
    """
    with Image.open(image_path).convert("RGB") as image:
        image = image.resize(input_size)
        im_arr = np.fromstring(image.tobytes(), dtype=np.uint8) / 255.0
        try:
            im_arr = im_arr.reshape((image.size[1], image.size[0], 3))
        except ValueError as e:
            im_arr = im_arr.reshape((image.size[1], image.size[0]))
            im_arr = np.stack((im_arr,)*3, axis=-1)
    return im_arr


def create_model(max_batch):
    search_space_updates = HyperparameterSearchSpaceUpdates()
    #TODO: this still runs out of memory and wastes resources
    search_space_updates.append(node_name="CreateImageDataLoader", hyperparameter="batch_size", log=False, \
                                value_range=[2, max_batch]) 
    # shutil.rmtree(save_output_to)
    autonet = AutoNetClassification(
                                    preset, \
                                    # hyperparameter_search_space_updates=search_space_updates, \
                                    min_workers=2, \
                                    # dataloader_worker=4, \
                                    # global_results_dir="results", \
                                    # keep_only_incumbent_checkpoints=False, \
                                    log_level="info", \
                                    budget_type="time", \
                                    # save_checkpoints=True, \
                                    result_logger_dir=save_output_to, \
                                    min_budget=200, \
                                    max_budget=600, \
                                    num_iterations=1, \
                                    # images_shape=[channels, input_size[0], input_size[1]], \
                                    optimizer = ["adam", "adamw", "sgd", "rmsprop"], \
                                    algorithm="hyperband", \
                                    optimize_metric="balanced_accuracy", \
                                    additional_metrics=["pac_metric"], \
                                    lr_scheduler=["cosine_annealing", "cyclic", "step", "adapt", "plateau", "alternating_cosine", "exponential"], \
                                    networks=['mlpnet', 'shapedmlpnet', 'resnet', 'shapedresnet'], #, 'densenet_flexible', 'resnet', 'resnet152', 'darts'], \
                                    use_tensorboard_logger=True, \
                                    cuda=True \
                                    )
    return autonet


if __name__ == "__main__":
    if args.prepare_data:
        prepare_dataset(chexpertpath)
        copy = True
    else:
        copy = False
    #Get the dataset (this could be yielded/batched)
    images, labels, le  = get_data_references(filter="train_" , copy_to = os.path.join(work_location, "train"), copy=copy)
    
    model = create_model(max_batch=int(len(labels)/20)) #Lipschitz magical number
    X = []
    for i, im in enumerate(images):
        image = jpg_image_to_array(im)
        X.append(image)
    #autopytorch format
    X = [np.asarray(x).flatten() for x in X]
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, labels, test_size=0.1, random_state=9, shuffle=True)
    results_fit = model.fit(\
                        X_train=X_train, Y_train=y_train, \
                        X_valid=X_test, Y_valid=y_test, \
                        refit=True, \
                        )
    with open("{}/results_fit.json".format(save_output_to), "w") as file:
        json.dump(results_fit, file)
