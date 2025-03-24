import json
import sys
sys.path.append('.')
import argparse
import logging
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from evaluate_metric import TransformationLoss, ClassificationLoss
from dataset import ThreeDLoader
from benchmark_utils import set_seed, icp_refine
from utils.timer import Timer
from Initial_matching import Matcher
from initial_matching_plus import Matcher_plus
from Correspondence_regenerate_v2 import Regenerator
from Tranformation_estimaton import Estimator
set_seed()
from utils.SE3 import transform
import visualization
import open3d


def read_trajectory_info(filename, dim=6):
    """
    Function that reads the trajectory information saved in the 3DMatch/Redwood format to a numpy array.
    Information file contains the variance-covariance matrix of the transformation paramaters.
    Format specification can be found at http://redwood-data.org/indoor/fileformat.html

    Args:
    filename (str): path to the '.txt' file containing the trajectory information data
    dim (int): dimension of the transformation matrix (4x4 for 3D data)

    Returns:
    n_frame (int): number of fragments in the scene
    cov_matrix (numpy array): covariance matrix of the transformation matrices for n pairs[n,dim, dim]
    """

    with open(filename) as fid:
        contents = fid.readlines()
    n_pairs = len(contents) // 7
    assert (len(contents) == 7 * n_pairs)
    info_list = []
    n_frame = 0

    for i in range(n_pairs):
        frame_idx0, frame_idx1, n_frame = [int(item) for item in contents[i * 7].strip().split()]
        info_matrix = np.concatenate(
            [np.fromstring(item, sep='\t').reshape(1, -1) for item in contents[i * 7 + 1:i * 7 + 7]], axis=0)
        info_list.append(info_matrix)

    cov_matrix = np.asarray(info_list, dtype=np.float).reshape(-1, dim, dim)

    return n_frame, cov_matrix

def eval_3DMatch_scene(loader, matcher, regenerator, estimator, trans_evaluator, cls_evaluator, scene, scene_ind, config, use_icp, gt_traj_cov):
    """
    Evaluate our model on 3DMatch testset [scene]
    """
    num_pair = loader.__len__()

    stats = np.zeros([num_pair, 20])
    data_timer, model_timer = Timer(), Timer()
    with torch.no_grad():
        error_pair = []
        fall_idx = []
        # for i in bad_list:
        for i in tqdm(range(num_pair)):
            # print(f"编号：{i}")
            #################################
            # 1. load data
            #################################
            data_timer.tic()
            src_keypts, tgt_keypts, src_features, tgt_features, gt_trans = loader.get_data(i)
            data_time = data_timer.toc()

            #################################
            # 2. match descriptor and compute rigid transformation
            #################################
            model_timer.tic()
            time1, time2, time3 = Timer(), Timer(), Timer()
            time1.tic()
            pred_trans, src_keypts_corr_filtered, tgt_keypts_corr_filtered, src_keypts_corr, tgt_keypts_corr, src_desc_corr_final, tgt_desc_corr_final = matcher.estimator(src_keypts, tgt_keypts, src_features, tgt_features, gt_trans)
            time1 = time1.toc()
            time2.tic()
            src_keypts_corr_final = src_keypts_corr_filtered
            tgt_keypts_corr_final = tgt_keypts_corr_filtered
            knn_num = 100
            sampling_num = 100

            '''fast approach'''
            # src_keypts_corr_final, tgt_keypts_corr_final, pred_trans = regenerator.regenerate(
            #     src_keypts_corr_final,
            #     tgt_keypts_corr_final,
            #     src_keypts,
            #     tgt_keypts,
            #     src_features,
            #     tgt_features,
            #     gt_trans,
            #     knn_num=20,
            #     sampling_num=500
            # )

            src_keypts_corr_final, tgt_keypts_corr_final, pred_trans = regenerator.regenerate(
                src_keypts_corr_final,
                tgt_keypts_corr_final,
                src_keypts,
                tgt_keypts,
                src_features,
                tgt_features,
                gt_trans,
                knn_num=100,
                sampling_num=100
            )

            src_keypts_corr_final, tgt_keypts_corr_final, pred_trans = regenerator.regenerate(
                src_keypts_corr_final,
                tgt_keypts_corr_final,
                src_keypts,
                tgt_keypts,
                src_features,
                tgt_features,
                gt_trans,
                knn_num=20,
                sampling_num=500
            )

            # src_keypts_corr_final, tgt_keypts_corr_final, pred_trans = regenerator.regenerate(
            #     src_keypts_corr_final,
            #     tgt_keypts_corr_final,
            #     src_keypts,
            #     tgt_keypts,
            #     src_features,
            #     tgt_features,
            #     gt_trans,
            #     knn_num=4,
            #     sampling_num=2500
            # )

            time2 = time2.toc()
            time3.tic()

            pred_trans,  pred_labels, src_corr_final, tgt_corr_final = estimator.estimator(src_keypts_corr_final, tgt_keypts_corr_final, src_keypts, tgt_keypts, gt_trans)
            # src_corr_final = src_keypts_corr_filtered
            # tgt_corr_final = tgt_keypts_corr_filtered
            time3 = time3.toc()
            # print(time1, time2, time3)

            model_time = model_timer.toc()
            #################################
            # 3. generate the ground-truth classification result
            #################################
            frag1_warp = transform(src_keypts_corr, gt_trans)  # 更改
            distance = torch.sum((frag1_warp - tgt_keypts_corr) ** 2, dim=-1) ** 0.5
            gt_labels = (distance < config.inlier_threshold).float()

            #################################
            # 4. evaluate result
            #################################
            recall, Re, Te, rmse = trans_evaluator(pred_trans, gt_trans, src_corr_final, tgt_corr_final, gt_traj_cov[i,:,:])
            class_stats = cls_evaluator(gt_trans, src_corr_final, tgt_corr_final, src_keypts_corr, tgt_keypts_corr)

            if recall == 0:
                fall_idx.append(i)

            #################################
            # 5. save the result
            #################################
            stats[i, 0] = float(recall / 100.0)                      # success
            stats[i, 1] = float(Re)                                  # Re (deg)
            stats[i, 2] = float(Te)                                  # Te (cm)
            stats[i, 3] = int(torch.sum(gt_labels))                  # input inlier number
            stats[i, 4] = float(torch.mean(gt_labels.float()))       # input inlier ratio
            stats[i, 5] = float(class_stats['output_inlier_number']) # output inlier number
            stats[i, 6] = float(class_stats['precision'])            # output inlier precision
            stats[i, 7] = float(class_stats['recall'])               # output inlier recall
            stats[i, 8] = float(class_stats['f1'])                   # output inlier f1 score
            stats[i, 9] = model_time
            stats[i, 10] = data_time
            stats[i, 11] = scene_ind
            stats[i, 12] = float(class_stats['feature_match_recall'])  # feature_match_recall
            stats[i, 13] = float(class_stats['feature_match_recall_0.1'])  # feature_match_recall
            stats[i, 14] = float(class_stats['feature_match_recall_0.01'])  # feature_match_recall
            stats[i, 15] = float(class_stats['IR_ratio'])  # IR_ratio
            stats[i, 16] = float(class_stats['INR'])  # INR
            stats[i, 17] = float(class_stats['NR'])  # NR
            stats[i, 19] = float(class_stats['inlier_nums'])  # NR

        print(fall_idx)

    return stats


def eval_3DMatch(config, use_icp):
    """
    Collect the evaluation results on each scene of 3DMatch testset, write the result to a .log file.
    """
    scene_list = [
        '7-scenes-redkitchen',
        'sun3d-home_at-home_at_scan1_2013_jan_1',
        'sun3d-home_md-home_md_scan9_2012_sep_30',
        'sun3d-hotel_uc-scan3',
        'sun3d-hotel_umd-maryland_hotel1',
        'sun3d-hotel_umd-maryland_hotel3',
        'sun3d-mit_76_studyroom-76-1studyroom2',
        'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
    ]
    all_stats = {}
    for scene_ind, scene in enumerate(scene_list):
        loader = ThreeDLoader(
            root=config.data_path,
            descriptor=config.descriptor,
            inlier_threshold=config.inlier_threshold,
            num_node=config.num_node,
            use_mutual=config.use_mutual,
            select_scene=scene,
        )

        matcher = Matcher_plus(
            inlier_threshold=0.1,
            num_node="all",
            use_mutual=False,
            d_thre=0.1,
            num_iterations=10,
            ratio=0.2,
            nms_radius=0.1,
            max_points=8000,
            k1=20,
            k2=10,
            FS_TCD_thre=0.05,
            relax_match_num=100,
            NS_by_IC=50
        )

        regenerator = Regenerator()
        estimator = Estimator()
        trans_evaluator = TransformationLoss(re_thre=config.re_thre, te_thre=config.te_thre)
        cls_evaluator = ClassificationLoss(inlier_threshold=config.inlier_threshold)
        _, gt_traj_cov = read_trajectory_info(os.path.join(f"/home/zhaoguiyu/code/OverlapPredator-main/configs/benchmarks/{config.dataset}/{scene}", "gt.info"))


        scene_stats = eval_3DMatch_scene(loader, matcher, regenerator, estimator, trans_evaluator, cls_evaluator, scene, scene_ind, config, use_icp, gt_traj_cov)
        all_stats[scene] = scene_stats
    logging.info(f"Max memory allicated: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}GB")

    # result for each scene
    scene_vals = np.zeros([len(scene_list), 20])
    scene_ind = 0
    for scene, stats in all_stats.items():
        correct_pair = np.where(stats[:, 0] == 1)
        scene_vals[scene_ind] = stats.mean(0)
        # for Re and Te, we only average over the successfully matched pairs.
        scene_vals[scene_ind, 1] = stats[correct_pair].mean(0)[1]
        scene_vals[scene_ind, 2] = stats[correct_pair].mean(0)[2]
        logging.info(f"Scene {scene_ind}th:"
                     f" Reg Recall={scene_vals[scene_ind, 0] * 100:.2f}% "
                     f" Mean RE={scene_vals[scene_ind, 1]:.2f} "
                     f" Mean TE={scene_vals[scene_ind, 2]:.2f} "
                     f" Mean Precision={scene_vals[scene_ind, 6] * 100:.2f}% "
                     f" Mean Recall={scene_vals[scene_ind, 7] * 100:.2f}% "
                     f" Mean F1={scene_vals[scene_ind, 8] * 100:.2f}%"
                     f" Mean Feature_match_recall={scene_vals[scene_ind, 12] * 100:.2f}%"
                     f" Mean Feature_match_recall>10%={scene_vals[scene_ind, 13] * 100:.2f}%"
                     f" Mean Feature_match_recall>1%={scene_vals[scene_ind, 14] * 100:.2f}%"
                     )
        scene_ind += 1

    # scene level average
    average = scene_vals.mean(0)
    logging.info(f"All {len(scene_list)} scenes, Mean Reg Recall={average[0] * 100:.2f}%, Mean Re={average[1]:.2f}, Mean Te={average[2]:.2f}， Mean FMR={average[12]* 100:.2f}")
    logging.info(f"\tMean FMR={average[12]* 100:.2f}, Mean FMR>10%={average[13]* 100:.2f}, Mean FMR>1%={average[14]* 100:.2f}")
    logging.info(f"\tInput:  Mean Inlier Num={average[3]:.2f}(ratio={average[4] * 100:.2f}%)")
    logging.info(f"\tOutput: Mean Inlier Num={average[5]:.2f}(precision={average[6] * 100:.2f}%, ratio={average[7] * 100:.2f}%, f1={average[8] * 100:.2f}%)")
    logging.info(f"\tcorrs: IR_ratio={average[15] * 100:.2f}%, INR={average[16] * 100:.2f}%, NR={average[17] * 100:.2f}%")
    logging.info(f"\tMean model time: {average[9]:.4f}s, Mean data time: {average[10]:.4f}s")

    # pair level average
    stats_list = [stats for _, stats in all_stats.items()]
    allpair_stats = np.concatenate(stats_list, axis=0)
    allpair_average = allpair_stats.mean(0)

    correct_pair_average = allpair_stats[allpair_stats[:, 0] == 1].mean(0)
    logging.info(f"*" * 40)
    logging.info(f"All {allpair_stats.shape[0]} pairs, Mean Reg Recall={allpair_average[0] * 100:.2f}%, Mean Re={correct_pair_average[1]:.2f}, Mean Te={correct_pair_average[2]:.2f}， Mean FMR={allpair_average[12]* 100:.2f}")
    logging.info(f"\tMean FMR={allpair_average[12]* 100:.2f}, Mean FMR>10%={allpair_average[13]* 100:.2f}, Mean FMR>1%={allpair_average[14]* 100:.2f}")
    logging.info(f"\tInput:  Mean Inlier Num={allpair_average[3]:.2f}(ratio={allpair_average[4] * 100:.2f}%)")
    logging.info(f"\tOutput: Mean Inlier Num={allpair_average[5]:.2f}(precision={allpair_average[6] * 100:.2f}%, ratio={allpair_average[7] * 100:.2f}%, f1={allpair_average[8] * 100:.2f}%)")
    logging.info(f"\tcorrs: IR_ratio={allpair_average[15] * 100:.2f}%, INR={allpair_average[16] * 100:.2f}%, IN={allpair_average[19]:.2f}")
    logging.info(f"\tMean model time: {allpair_average[9]:.4f}s, Mean data time: {allpair_average[10]:.4f}s")

    all_stats_npy = np.concatenate([v for k, v in all_stats.items()], axis=0)
    return all_stats_npy


if __name__ == '__main__':
    from config import str2bool

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='', type=str, help='snapshot dir')
    parser.add_argument('--solver', default='SVD', type=str, choices=['SVD', 'RANSAC'])
    parser.add_argument('--use_icp', default=False, type=str2bool)
    parser.add_argument('--save_npy', default=False, type=str2bool)
    args = parser.parse_args()

    config_path = args.config_path
    config = json.load(open(config_path, 'r'))
    config = edict(config)

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_Devices
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    log_filename = f'logs/{config.dataset}-{config.descriptor}.log'
    logging.basicConfig(level=logging.INFO,
                        filename=log_filename,
                        filemode='a',
                        format="")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # evaluate on the test set
    stats = eval_3DMatch(config, args.use_icp)
    if args.save_npy:
        save_path = log_filename.replace('.log', '.npy')
        np.save(save_path, stats)
        print(f"Save the stats in {save_path}")
