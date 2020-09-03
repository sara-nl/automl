import argparse
import numpy as np
from tensorflow.keras.datasets import mnist
from ray.tune.integration.keras import TuneReporterCallback
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from auto_chexpert import *

def train_chexpert(config):
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


if __name__ == "__main__":
    ray.init(num_cpus=4 if args.smoke_test else None)
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        metric="mean_accuracy",
        mode="max",
        max_t=400,
        grace_period=20)

    tune.run(
        train_chexpert,
        name="exp",
        scheduler=sched,
        stop={
            "mean_accuracy": 0.99,
            "training_iteration": 5 if args.smoke_test else 300
        },
        num_samples=10,
        resources_per_trial={
            "cpu": 2,
            "gpu": 0
        },
        config={
            "threads": 2,
            "lr": tune.sample_from(lambda spec: np.random.uniform(0.001, 0.1))
        })
