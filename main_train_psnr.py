import argparse
import logging
import math
import os
import os.path
import random
import time

import numpy as np
import torch
from clearml import Dataset, Task
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.select_dataset import define_Dataset
from models.select_model import define_Model
from utils import utils_image as util
from utils import utils_logger
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

"""
# --------------------------------------------
# training code for MSRResNet
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
"""


def main(json_path="options/train_msrresnet_psnr.json"):

    """
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    """
    task_name = "Test run #2 for KAIR SwinIR Optical 5m to 2.5m new clearml dataset"
    print(os.environ.get("LOCAL_RANK", 0))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opt", type=str, default=json_path, help="Path to option JSON file."
    )
    parser.add_argument("--launcher", default="pytorch", help="job launcher")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--dist", default=False)
    clearml_project = "Super Resolution"
    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt["dist"] = parser.parse_args().dist
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print("Yeah")
        task = Task.init(
            project_name=clearml_project, task_name=task_name, continue_last_task=True
        )
        task.connect(opt)
    else:
        task = Task.get_task(project_name=clearml_project, task_name=task_name)

    # ----------------------------------------
    # get clearml dataset
    # ----------------------------------------

    clearml_name = "Eastern Township Super-Resolution RGB C"
    dataset_location = os.path.abspath(
        "./donnees"
    )  #'/home/horoma/data/70.SuperResolution/ClearMLDataset/'

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        if (
            opt.get("cache_path", None) is not None
        ):  # The default verification should be in the config validation module
            os.environ["CLEARML_CACHE_DIR"] = os.path.abspath(
                opt["cache_path"]
            )  # This seems hacky but works as intended

        #       clearml_dataset = Dataset.get(dataset_id="5f8672cca9844d1990bffaa388191e9b", only_published=True)
        clearml_dataset = Dataset.get(
            dataset_project=clearml_project,
            dataset_name=clearml_name,
            only_published=True,
        )
        clearml_dataset.verify_dataset_hash(verbose=False)
        dataset_location = clearml_dataset.get_local_copy()
        with open("dataset_location.txt", "w") as fi:
            fi.write(dataset_location)
    else:
        while not os.path.isfile("dataset_location.txt"):
            time.sleep(60)
        with open("dataset_location.txt", "r") as fi:
            dataset_location = (fi.read()).strip("\n")

    #    if not os.path.exists(dataset_location + 'train'):

    #        dataset = Dataset.get(
    #            dataset_id=None,
    #            dataset_project=clearml_project,
    #            dataset_name=clearml_name,
    #            only_completed=True
    #        )
    #        dataset.get_mutable_local_copy(dataset_location)

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt["dist"]:
        init_dist("pytorch")
    opt["rank"], opt["world_size"] = get_dist_info()

    if opt["rank"] == 0:
        util.mkdirs(
            (path for key, path in opt["path"].items() if "pretrained" not in key)
        )

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option.find_last_checkpoint(
        opt["path"]["models"], net_type="G"
    )
    init_iter_E, init_path_E = option.find_last_checkpoint(
        opt["path"]["models"], net_type="E"
    )
    opt["path"]["pretrained_netG"] = init_path_G
    opt["path"]["pretrained_netE"] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(
        opt["path"]["models"], net_type="optimizerG"
    )
    opt["path"]["pretrained_optimizerG"] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)
    # ----------------------------------------
    # add clearml dataset location prefix
    # ----------------------------------------
    opt["datasets"]["train"]["dataroot_H"] = (
        dataset_location + opt["datasets"]["train"]["dataroot_H"]
    )
    opt["datasets"]["train"]["dataroot_L"] = (
        dataset_location + opt["datasets"]["train"]["dataroot_L"]
    )
    opt["datasets"]["test"]["dataroot_H"] = (
        dataset_location + opt["datasets"]["test"]["dataroot_H"]
    )
    opt["datasets"]["test"]["dataroot_L"] = (
        dataset_location + opt["datasets"]["test"]["dataroot_L"]
    )

    border = opt["scale"]
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt["rank"] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt["rank"] == 0:
        logger_name = "train"
        utils_logger.logger_info(
            logger_name, os.path.join(opt["path"]["log"], logger_name + ".log")
        )
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt["train"]["manual_seed"]
    if seed is None:
        seed = random.randint(1, 10000)
    print("Random seed: {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    """
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    """

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = define_Dataset(dataset_opt)
            train_size = int(
                math.ceil(len(train_set) / dataset_opt["dataloader_batch_size"])
            )
            if opt["rank"] == 0:
                logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size
                    )
                )
            if opt["dist"]:
                train_sampler = DistributedSampler(
                    train_set,
                    shuffle=dataset_opt["dataloader_shuffle"],
                    drop_last=True,
                    seed=seed,
                )
                train_loader = DataLoader(
                    train_set,
                    batch_size=dataset_opt["dataloader_batch_size"] // opt["num_gpu"],
                    shuffle=False,
                    num_workers=dataset_opt["dataloader_num_workers"] // opt["num_gpu"],
                    drop_last=True,
                    pin_memory=True,
                    sampler=train_sampler,
                )
            else:
                train_loader = DataLoader(
                    train_set,
                    batch_size=dataset_opt["dataloader_batch_size"],
                    shuffle=dataset_opt["dataloader_shuffle"],
                    num_workers=dataset_opt["dataloader_num_workers"],
                    drop_last=True,
                    pin_memory=True,
                )

        elif phase == "test":
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(
                test_set,
                batch_size=1,
                shuffle=False,
                num_workers=1,
                drop_last=False,
                pin_memory=True,
            )
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    """
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    """

    model = define_Model(opt)
    model.init_train()
    if opt["rank"] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    """
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    """
    for epoch in range(opt["train"]["number of epochs"]):  # keep running
        if opt["dist"]:
            train_sampler.set_epoch(epoch)

        for i, train_data in enumerate(train_loader):

            current_step += 1

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 4) training information
            # -------------------------------
            if (
                current_step % opt["train"]["checkpoint_print"] == 0
                and opt["rank"] == 0
            ):
                logs = model.current_log()  # such as loss
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model.current_learning_rate()
                )
                for k, v in logs.items():  # merge log information into message
                    message += "{:s}: {:.3e} ".format(k, v)
                    task.get_logger().report_scalar(
                        k,
                        "worker {:02d}".format(opt["rank"]),
                        value=v,
                        iteration=current_step,
                    )
                logger.info(message)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt["train"]["checkpoint_save"] == 0 and opt["rank"] == 0:
                logger.info("Saving the model.")
                model.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt["train"]["checkpoint_test"] == 0 and opt["rank"] == 0:

                avg_psnr = 0.0
                idx = 0

                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data["L_path"][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    img_dir = os.path.join(opt["path"]["images"], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(test_data)
                    model.test()

                    visuals = model.current_visuals()
                    E_img = util.tensor2uint(visuals["E"])
                    H_img = util.tensor2uint(visuals["H"])

                    # -----------------------
                    # save estimated image E
                    # -----------------------
                    save_img_path = os.path.join(
                        img_dir, "{:s}_{:d}.png".format(img_name, current_step)
                    )
                    util.imsave(E_img, save_img_path)
                    task.get_logger().report_image(
                        img_name, "Generated Image", iteration=current_step, image=E_img
                    )

                    task.get_logger().report_image(
                        img_name, "True Image", iteration=current_step, image=H_img
                    )

                    # -----------------------
                    # calculate PSNR
                    # -----------------------
                    current_psnr = util.calculate_psnr(E_img, H_img, border=border)

                    logger.info(
                        "{:->4d}--> {:>10s} | {:<4.2f}dB".format(
                            idx, image_name_ext, current_psnr
                        )
                    )

                    avg_psnr += current_psnr

                avg_psnr = avg_psnr / idx

                # testing log
                logger.info(
                    "<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB\n".format(
                        epoch, current_step, avg_psnr
                    )
                )
                task.get_logger().report_scalar(
                    "Test average PSNR",
                    "worker {:02d}".format(opt["rank"]),
                    value=avg_psnr,
                    iteration=current_step,
                )


if __name__ == "__main__":
    main()
