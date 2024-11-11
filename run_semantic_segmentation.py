import argparse
import os

import pandas as pd
import pickle
from autogluon.multimodal import MultiModalPredictor
import torch

def get_default_training_setting(dataset_name):
    validation_metric = "iou"
    loss = "structure_loss"
    max_epoch = 20
    lr = 1e-4

    if dataset_name == "SBU-shadow":
        validation_metric = "ber"
        loss = "balanced_bce"
        max_epoch = 10

    elif dataset_name == "polyp":
        validation_metric = "sm"
 
    elif dataset_name == "gan-generated":
        validation_metric = "iou" #iou for semnantic segmentation
        max_epoch = 10 #15#100
        lr = 3e-4
        loss = "structure_loss"
        #loss = "mask2former_loss" #remove this for semantic segmentation

    return validation_metric, loss, max_epoch, lr


def expand_path(df, dataset_dir):
    for col in ["image", "label"]:
        df[col] = df[col].apply(lambda ele: os.path.join(dataset_dir, ele))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script support converting voc format xmls to coco format json")
    parser.add_argument(
        "--task",
        type=str,
        default="leaf_disease_segmentation",
        choices=["real", "gan-generated"],
    )
    parser.add_argument("--seed", type=int, default=42686693)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--expert_num", type=int, default=8)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--dataset_dir", type=str, default="datasets/")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--data_name", type=str, default="test1_class1")
    parser.add_argument("--ckpt_path", type=str, default="outputs", help="Checkpoint path.")
    parser.add_argument("--per_gpu_batch_size", type=int, default=1, help="The batch size for each GPU.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="The effective batch size. If batch_size > per_gpu_batch_size * num_gpus, gradient accumulation would be used.",
    )
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    dataset_name = args.task
    #dataset_dir = os.path.join(f"datasets/{dataset_name}", dataset_name)
    dataset_dir = os.path.join(args.dataset_dir, dataset_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # prepare dataframes
    #train_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"train_defect_class1.csv")), dataset_dir)
    #val_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"test_defect_class1.csv")), dataset_dir)
    train_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"train_{args.data_name}.csv")), dataset_dir)
    val_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"test_{args.data_name}.csv")), dataset_dir)

    # get the validation metric
    validation_metric, loss, max_epoch, lr = get_default_training_setting(dataset_name)

    hyperparameters = {}
    hyperparameters.update(
        {
            "optimization.lora.r": args.rank,
            "optimization.efficient_finetune": "conv_lora",
            "optimization.lora.conv_lora_expert_num": args.expert_num,
            "env.num_gpus": args.num_gpus,
            "optimization.loss_function": loss,
            "optimization.max_epochs": max_epoch,
            "optimization.learning_rate": lr,
            "env.per_gpu_batch_size": args.per_gpu_batch_size,
            "env.batch_size": args.batch_size,
        }
    )
    torch.cuda.empty_cache()
    if args.eval:  # load a checkpoint for evaluation
        predictor = MultiModalPredictor.load(args.ckpt_path)
    else:  # training
        
        predictor = MultiModalPredictor(
            problem_type="semantic_segmentation", 
            validation_metric=validation_metric,
            eval_metric=validation_metric,
            hyperparameters=hyperparameters,
            label="label",
        )
        
        #predictor = MultiModalPredictor.load(args.ckpt_path)
        '''
        #for multiclass gan and real data
        predictor = MultiModalPredictor(
            problem_type = "semantic_segmentation", 
            validation_metric = validation_metric, #multiclass for multiclass gan data
            eval_metric = validation_metric,
            hyperparameters = hyperparameters,
            num_classes = 5,
            classes = [0, 1, 2, 3, 4],
            label = "label",
        )
        '''
        predictor.fit(train_data=train_df, tuning_data=val_df, seed=args.seed)

    # evaluation
    #metric_file = os.path.join(args.output_dir, "metrics_defect_class1.txt")
    #out_file = os.path.join(args.output_dir, "output_defect_class1.pkl") 
    metric_file = os.path.join(args.output_dir, f"metrics_{args.data_name}.txt")
    out_file = os.path.join(args.output_dir, f"output_{args.data_name}.pkl") 
    f = open(metric_file, "a")
    if dataset_name in ["gan-generated", "real"]:
        #test_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"test_defect_class1.csv")), dataset_dir)
        test_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"train_{args.data_name}.csv")), dataset_dir)
        eval_metrics = ["iou"]
        #eval_metrics = ["accuracy","iou"]

        res, output = predictor.evaluate(test_df, metrics=eval_metrics, return_pred=True)
        print(f"Evaluation results for test dataset {dataset_name}: ", res)
        f.write(f"Evaluation results for test dataset {dataset_name}: {res} \n")
        
        with open(out_file, "wb") as f:
            pickle.dump(output, f)
    '''    
    elif dataset_name in ["polyp", "camo_sem_seg"]:
        if dataset_name == "polyp":
            test_datasets = ["CVC-ClinicDB", "Kvasir"]
        elif dataset_name == "camo_sem_seg":
            test_datasets = ["CAMO"]
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}.")
        for per_dataset in test_datasets:
            test_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"test_{per_dataset}.csv")), dataset_dir)
            res = predictor.evaluate(test_df, metrics=["sm", "fm", "em", "mae"])
            print(f"Evaluation results for test dataset {per_dataset}: ", res)
            f.write(f"Evaluation results for test dataset {per_dataset}: {res} \n")
    '''
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}.")

    f.close()
