import argparse
import glob
import os
from clearml import Dataset, Task, Logger
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd
import requests
import torch
from models.network_swinir import SwinIR as net
from utils import utils_image as util
from utils import utils_option as option


def main():
    clearml_project = "Super Resolution"
    task_name = "Classification run for KAIR SwinIR Optical 5m to 2.5m new clearml dataset"
    task = Task.init(project_name=clearml_project,task_name=task_name,task_type=Task.TaskTypes.testing,reuse_last_task_id=False)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opt",
        type=str,
        default="./options/swinir/train_swinir_sr_lightweight.json",
        help="Path to option JSON file.",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="lightweight_sr",
        help="classical_sr, lightweight_sr, real_sr, " "gray_dn, color_dn, jpeg_car",
    )
    parser.add_argument(
        "--scale", type=int, default=0, help="scale factor: 1, 2, 3, 4, 8"
    )  # 1 for dn and jpeg car
    parser.add_argument("--noise", type=int, default=15, help="noise level: 15, 25, 50")
    parser.add_argument(
        "--jpeg", type=int, default=40, help="scale factor: 10, 20, 30, 40"
    )
    parser.add_argument(
        "--training_patch_size",
        type=int,
        default=0,
        help="patch size used in training SwinIR. "
        "Just used to differentiate two different settings in Table 2 of the paper. "
        "Images are NOT tested patch by patch.",
    )
    parser.add_argument(
        "--large_model",
        action="store_true",
        help="use large model, only provided for real image sr",
    )
    parser.add_argument("--model_path", type=str, default="from config")
    parser.add_argument(
        "--folder_lq",
        type=str,
        default=None,
        help="input low-quality test image folder",
    )
    parser.add_argument(
        "--folder_gt",
        type=str,
        default=None,
        help="input ground-truth test image folder",
    )
    parser.add_argument(
        "--tile",
        type=int,
        default=None,
        help="Tile size, None for no tile during testing (testing as a whole)",
    )
    parser.add_argument(
        "--tile_overlap", type=int, default=32, help="Overlapping of different tiles"
    )
    args = parser.parse_args()
    opt = option.parse(parser.parse_args().opt, is_train=True)
    init_iter_E, init_path_E = option.find_last_checkpoint(
        opt["path"]["models"], net_type="E"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # set up model
    if os.path.exists(args.model_path):
        print(f"loading model from {args.model_path}")
    elif args.model_path == "from config":
        print("loading model from config file")
        args.model_path = init_path_E
    else:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        url = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}".format(
            os.path.basename(args.model_path)
        )
        r = requests.get(url, allow_redirects=True)
        print(f"downloading model {args.model_path}")
        open(args.model_path, "wb").write(r.content)
    if args.scale == 0:
        args.scale = opt["scale"]
    if args.training_patch_size == 0:
        args.training_patch_size = opt["netG"]["img_size"]
    with open("dataset_location.txt", "r") as fi:
        dataset_location = (fi.read()).strip("\n")
    if args.folder_lq is None:
        args.folder_lq = dataset_location + (
            opt["datasets"]["test"]["dataroot_L"]
        ).replace("val", "test")
    if args.folder_gt is None:
        args.folder_gt = dataset_location + (
            opt["datasets"]["test"]["dataroot_H"]
        ).replace("val", "test")
    stats = open("Stats_" + task_name + ".txt", "w")

    model = define_model(args, opt)
    model.eval()
    model = model.to(device)

    # setup folder and path
    folder, save_dir, border, window_size = setup(args)
    os.makedirs(save_dir, exist_ok=True)
    test_results = OrderedDict()
    test_results["image"] = []
    test_results["psnr"] = []
    test_results["ssim"] = []
    test_results["psnr_y"] = []
    test_results["ssim_y"] = []
    test_results["psnr_b"] = []
    psnr, ssim, psnr_y, ssim_y, psnr_b = 0, 0, 0, 0, 0

    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, "*")))):
        # read image
        imgname, img_lq, img_gt = get_image_pair(
            args, path
        )  # image to HWC-BGR, float32
        #        print (args.folder_lq,args.folder_gt,imgname, img_lq.shape, img_gt.shape,folder)
        img_lq = np.transpose(
            img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1)
        )  # HCW-BGR to CHW-RGB
        img_lq = (
            torch.from_numpy(img_lq).float().unsqueeze(0).to(device)
        )  # CHW-RGB to NCHW-RGB

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[
                :, :, : h_old + h_pad, :
            ]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[
                :, :, :, : w_old + w_pad
            ]
            output = test(img_lq, model, args, window_size)
            output = output[..., : h_old * args.scale, : w_old * args.scale]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(
                output[[2, 1, 0], :, :], (1, 2, 0)
            )  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        cv2.imwrite(f"{save_dir}/{imgname}.tif", output)

        task.get_logger().report_image(
            imgname, "Generated Image", local_path=f"{save_dir}/{imgname}.tif"
        )

        task.get_logger().report_image(
            imgname,
            "Reference Image",
            local_path=path
            #            image=np.transpose(img_gt if img_gt.shape[2] == 1 else img_gt[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        )

        # evaluate psnr/ssim/psnr_b
        if img_gt is not None:
            img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
            img_gt = img_gt[: h_old * args.scale, : w_old * args.scale, ...]  # crop gt
            img_gt = np.squeeze(img_gt)
            psnr = util.calculate_psnr(output, img_gt, border=border)
            ssim = util.calculate_ssim(output, img_gt, border=border)
            test_results["image"].append(imgname)
            test_results["psnr"].append(psnr)
            test_results["ssim"].append(ssim)
            if img_gt.ndim == 3:  # RGB image
                output_y = util.bgr2ycbcr(output.astype(np.float32) / 255.0) * 255.0
                img_gt_y = util.bgr2ycbcr(img_gt.astype(np.float32) / 255.0) * 255.0
                psnr_y = util.calculate_psnr(output_y, img_gt_y, border=border)
                ssim_y = util.calculate_ssim(output_y, img_gt_y, border=border)
                test_results["psnr_y"].append(psnr_y)
                test_results["ssim_y"].append(ssim_y)
            if args.task in ["jpeg_car"]:
                psnr_b = util.calculate_psnrb(output, img_gt, border=border)
                test_results["psnr_b"].append(psnr_b)
            print(
                "Testing {:d} {:20s} - PSNR: {:.2f} dB; SSIM: {:.4f}; "
                "PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}; "
                "PSNR_B: {:.2f} dB.".format(
                    idx, imgname, psnr, ssim, psnr_y, ssim_y, psnr_b
                ),
                file=stats,
            )
        #            task.get_logger().report_scalar("Image",imgname)
        #            task.get_logger().report_scalar("PSNR",psnr)
        #            task.get_logger().report_scalar("SSIM",ssim)
        #            task.get_logger().report_scalar("PSNR_Y",psnr_y)
        #            task.get_logger().report_scalar("SSIM_Y",ssim_y)
        #            task.get_logger().report_scalar("PSNR_B",psnr_b)
        else:
            print("Testing {:d} {:20s}".format(idx, imgname), file=stats)

    # summarize psnr/ssim
    if img_gt is not None:
        ave_psnr = sum(test_results["psnr"]) / len(test_results["psnr"])
        ave_ssim = sum(test_results["ssim"]) / len(test_results["ssim"])
        print(
            "\n{} \n-- Average PSNR/SSIM(RGB): {:.2f} dB; {:.4f}".format(
                save_dir, ave_psnr, ave_ssim
            ),
            file=stats,
        )
        if img_gt.ndim == 3:
            ave_psnr_y = sum(test_results["psnr_y"]) / len(test_results["psnr_y"])
            ave_ssim_y = sum(test_results["ssim_y"]) / len(test_results["ssim_y"])
            print(
                "-- Average PSNR_Y/SSIM_Y: {:.2f} dB; {:.4f}".format(
                    ave_psnr_y, ave_ssim_y
                ),
                file=stats,
            )
        if args.task in ["jpeg_car"]:
            ave_psnr_b = sum(test_results["psnr_b"]) / len(test_results["psnr_b"])
            print("-- Average PSNR_B: {:.2f} dB".format(ave_psnr_b), file=stats)
    # log statistics to clearml
    if args.task in ["jpeg_car"]:
        table = pd.DataFrame(
            {
                "PSNR": test_results["psnr"],
                "SSIM": test_results["ssim"],
                "PSNR_Y": test_results["psnr_y"],
                "SSIM_Y": test_results["ssim_y"],
                "PSNR_B": test_results["psnr_b"],
            },
            index=test_results["image"],
        )
        table_summary = pd.DataFrame(
            {
                "Average PSNR": [ave_psnr],
                "Average SSIM": [ave_ssim],
                "Average PSNR_Y": [ave_psnr_y],
                "Average SSIM_Y": [ave_ssim_y],
                "Average PSNR_B": [ave_psnr_b],
            },
            index=["Averages"],
        )

    else:
        table = pd.DataFrame(
            {
                "PSNR": test_results["psnr"],
                "SSIM": test_results["ssim"],
                "PSNR_Y": test_results["psnr_y"],
                "SSIM_Y": test_results["ssim_y"],
            },
            index=test_results["image"]  
        )        
        table_summary = pd.DataFrame(
            {
                "Average PSNR": [ave_psnr],
                "Average SSIM": [ave_ssim],
                "Average PSNR_Y": [ave_psnr_y],
                "Average SSIM_Y": [ave_ssim_y],
            },
            index=["Averages"],
        )

    table.index.name = "image"
    table_summary.index.name = "Summary"
    task.get_logger().report_table(title='Statistics for estimated images',series='Standard metrics',iteration=0,table_plot=table)
    task.get_logger().report_table(title='Average statistics for estimated images',series='Standard metrics',iteration=0,table_plot=table_summary)


def define_model(args, opt):
    # 001 classical image sr
    if args.task in [
        "classical_sr",
        "lightweight_sr",
        "gray_dn",
        "color_dn",
        "jpeg_car",
    ]:
        param_key_g = "params"
    elif args.task == "real_sr":
        param_key_g = "params_ema"

    model = net(
        upscale=args.scale,
        in_chans=opt["netG"]["in_chans"],
        img_size=args.training_patch_size,
        window_size=opt["netG"]["window_size"],
        img_range=opt["netG"]["img_range"],
        depths=opt["netG"]["depths"],
        embed_dim=opt["netG"]["embed_dim"],
        num_heads=opt["netG"]["num_heads"],
        mlp_ratio=opt["netG"]["mlp_ratio"],
        upsampler=opt["netG"]["upsampler"],
        resi_connection=opt["netG"]["resi_connection"],
    )

    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(
        pretrained_model[param_key_g]
        if param_key_g in pretrained_model.keys()
        else pretrained_model,
        strict=True,
    )

    return model


def setup(args):
    # 001 classical image sr/ 002 lightweight image sr
    if args.task in ["classical_sr", "lightweight_sr"]:
        save_dir = f"results/swinir_{args.task}_x{args.scale}"
        folder = args.folder_gt
        border = args.scale
        window_size = 8

    # 003 real-world image sr
    elif args.task in ["real_sr"]:
        save_dir = f"results/swinir_{args.task}_x{args.scale}"
        if args.large_model:
            save_dir += "_large"
        folder = args.folder_lq
        border = 0
        window_size = 8

    # 004 grayscale image denoising/ 005 color image denoising
    elif args.task in ["gray_dn", "color_dn"]:
        save_dir = f"results/swinir_{args.task}_noise{args.noise}"
        folder = args.folder_gt
        border = 0
        window_size = 8

    # 006 JPEG compression artifact reduction
    elif args.task in ["jpeg_car"]:
        save_dir = f"results/swinir_{args.task}_jpeg{args.jpeg}"
        folder = args.folder_gt
        border = 0
        window_size = 7

    return folder, save_dir, border, window_size


def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    # 001 classical image sr/ 002 lightweight image sr (load lq-gt image pairs)
    if args.task in ["classical_sr", "lightweight_sr"]:
        img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        img_lq = (
            cv2.imread(f"{args.folder_lq}/{imgname}{imgext}", cv2.IMREAD_COLOR).astype(
                np.float32
            )
            / 255.0
        )

    # 003 real-world image sr (load lq image only)
    elif args.task in ["real_sr"]:
        img_gt = None
        img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

    # 004 grayscale image denoising (load gt image and generate lq image on-the-fly)
    elif args.task in ["gray_dn"]:
        img_gt = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        np.random.seed(seed=0)
        img_lq = img_gt + np.random.normal(0, args.noise / 255.0, img_gt.shape)
        img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = np.expand_dims(img_lq, axis=2)

    # 005 color image denoising (load gt image and generate lq image on-the-fly)
    elif args.task in ["color_dn"]:
        img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        np.random.seed(seed=0)
        img_lq = img_gt + np.random.normal(0, args.noise / 255.0, img_gt.shape)

    # 006 JPEG compression artifact reduction (load gt image and generate lq image on-the-fly)
    elif args.task in ["jpeg_car"]:
        img_gt = cv2.imread(path, 0)
        result, encimg = cv2.imencode(
            ".jpg", img_gt, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg]
        )
        img_lq = cv2.imdecode(encimg, 0)
        img_gt = np.expand_dims(img_gt, axis=2).astype(np.float32) / 255.0
        img_lq = np.expand_dims(img_lq, axis=2).astype(np.float32) / 255.0

    return imgname, img_lq, img_gt


def test(img_lq, model, args, window_size):
    if args.tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
        E = torch.zeros(b, c, h * sf, w * sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx : h_idx + tile, w_idx : w_idx + tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[
                    ...,
                    h_idx * sf : (h_idx + tile) * sf,
                    w_idx * sf : (w_idx + tile) * sf,
                ].add_(out_patch)
                W[
                    ...,
                    h_idx * sf : (h_idx + tile) * sf,
                    w_idx * sf : (w_idx + tile) * sf,
                ].add_(out_patch_mask)
        output = E.div_(W)

    return output


if __name__ == "__main__":
    main()
