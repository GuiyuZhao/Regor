import torch
from common import knn, rigid_transform_3d
from utils.SE3 import transform
import numpy as np
import visualization
import open3d


class Estimator():
    def __init__(self,
                 inlier_threshold=0.10,
                 num_iterations=10,
                 num_node='all'
                 ):
        self.inlier_threshold = inlier_threshold
        self.Debug = False
        self.num_node = num_node
        self.num_iterations = num_iterations  # maximum iteration of power iteration algorithm
        self.gt_trans = 0

    def post_refinement(self, initial_trans, src_keypts, tgt_keypts, src_pts, tgt_pts, it_num, weights=None):
        """
        Perform post refinement using the initial transformation matrix, only adopted during testing.
        Input
            - initial_trans: [bs, 4, 4]
            - src_keypts:    [bs, num_corr, 3]
            - tgt_keypts:    [bs, num_corr, 3]
            - weights:       [bs, num_corr]
        Output:
            - final_trans:   [bs, 4, 4]
        """
        assert initial_trans.shape[0] == 1
        inlier_threshold = 1.2

        # inlier_threshold_list = [self.inlier_threshold] * it_num

        if self.inlier_threshold == 0.10:  # for 3DMatch
            inlier_threshold_list = [0.10] * it_num
        else:  # for KITTI
            inlier_threshold_list = [1.2] * it_num

        previous_inlier_num = 0
        for inlier_threshold in inlier_threshold_list:
            warped_src_keypts = transform(src_keypts, initial_trans)

            L2_dis = torch.norm(warped_src_keypts - tgt_keypts, dim=-1)
            pred_inlier = (L2_dis < inlier_threshold)[0]  # assume bs = 1
            inlier_num = torch.sum(pred_inlier)
            if abs(int(inlier_num - previous_inlier_num)) < 1:
                break
            else:
                previous_inlier_num = inlier_num
            initial_trans = rigid_transform_3d(
                A=src_keypts[:, pred_inlier, :],
                B=tgt_keypts[:, pred_inlier, :],
                ## https://link.springer.com/article/10.1007/s10589-014-9643-2
                # weights=None,
                weights=1 / (1 + (L2_dis / inlier_threshold) ** 2)[:, pred_inlier],
                # weights=((1-L2_dis/inlier_threshold)**2)[:, pred_inlier]
            )
        return initial_trans

    # 提升了1%，但是速度下降了
    def post_refinement_points(self, initial_trans, src_points, tgt_points, it_num, weights=None):
        """
        Perform post refinement using the initial transformation matrix, only adopted during testing.
        Input
            - initial_trans: [bs, 4, 4]
            - src_keypts:    [bs, num_corr, 3]
            - tgt_keypts:    [bs, num_corr, 3]
            - weights:       [bs, num_corr]
        Output:
            - final_trans:   [bs, 4, 4]
        """
        N_src = src_points.shape[1]
        N_tgt = tgt_points.shape[1]
        if self.num_node == 'all':
            src_sel_ind = np.arange(N_src)
            tgt_sel_ind = np.arange(N_tgt)
        else:
            src_sel_ind = np.random.choice(N_src, self.num_node)
            tgt_sel_ind = np.random.choice(N_tgt, self.num_node)
        src_points = src_points[:, src_sel_ind, :]
        tgt_points = tgt_points[:, tgt_sel_ind, :]

        assert initial_trans.shape[0] == 1
        inlier_threshold = 1.2

        # inlier_threshold_list = [self.inlier_threshold] * it_num

        if self.inlier_threshold == 0.10:  # for 3DMatch
            inlier_threshold_list = [0.10] * it_num
        else:  # for KITTI
            inlier_threshold_list = [1.2] * it_num

        previous_inlier_num = 0
        for inlier_threshold in inlier_threshold_list:
            warped_src_keypts = transform(src_points, initial_trans)
            distance = torch.norm(warped_src_keypts[:, :, None, :] - tgt_points[:, None, :, :], dim=-1)
            min_distance, min_idx = torch.min(distance.squeeze(0), dim=1)
            min_distance_mask = min_distance < inlier_threshold
            row_indices = torch.arange(0, min_idx.shape[0])[min_distance_mask].cuda()
            min_idx_select = min_idx[min_distance_mask]
            col_indices = min_idx_select
            inlier_num = min_idx_select.shape[0]

            if abs(int(inlier_num - previous_inlier_num)) < 1:
                break
            else:
                previous_inlier_num = inlier_num
            initial_trans = rigid_transform_3d(
                A=src_points[:, row_indices, :],
                B=tgt_points[:, col_indices, :],
                ## https://link.springer.com/article/10.1007/s10589-014-9643-2
                # weights=None,
                weights=1 / (1 + (distance[:, row_indices, col_indices] / inlier_threshold) ** 2),
                # weights=((1-L2_dis/inlier_threshold)**2)[:, pred_inlier]
            )

        src_corr_refine = src_points[:, row_indices, :]
        tgt_corr_refine = tgt_points[:, col_indices, :]
        return initial_trans, src_corr_refine, tgt_corr_refine

    def estimator(self, src_corr, tgt_corr, src_keypts, tgt_keypts, gt_trans):
        """
        Input:
            - src_corr: [bs, num_corr, 3]
            - tgt_corr: [bs, num_corr, 3]
        Output:
            - pred_trans:   [bs, 4, 4], the predicted transformation matrix
            - pred_trans:   [bs, num_corr], the predicted inlier/outlier label (0,1)
            - src_keypts_corr:  [bs, num_corr, 3], the source points in the matched correspondences
            - tgt_keypts_corr:  [bs, num_corr, 3], the target points in the matched correspondences
        """
        self.gt_trans = gt_trans
        #################################
        # use the proposed SC2-PCR to estimate the rigid transformation
        #################################
        pred_tran = rigid_transform_3d(src_corr, tgt_corr)
        # final_tran = pred_tran

        final_tran, src_corr_final, tgt_corr_final = self.post_refinement_points(pred_tran, src_keypts, tgt_keypts, 20)
        # final_tran = self.post_refinement(pred_tran,  src_corr, tgt_corr, src_keypts, tgt_keypts, 20)

        frag1_warp = transform(src_corr, pred_tran)
        distance = torch.sum((frag1_warp - tgt_corr) ** 2, dim=-1) ** 0.5
        pred_labels = (distance < self.inlier_threshold)
        # src_corr_final = src_corr[:, pred_labels[0], :]
        # tgt_corr_final = tgt_corr[:, pred_labels[0], :]

        return final_tran, pred_labels, src_corr_final, tgt_corr_final

    def estimator_with_or(self, src_corr, tgt_corr, src_keypts, tgt_keypts, gt_trans):
        """
        Input:
            - src_corr: [bs, num_corr, 3]
            - tgt_corr: [bs, num_corr, 3]
        Output:
            - pred_trans:   [bs, 4, 4], the predicted transformation matrix
            - pred_trans:   [bs, num_corr], the predicted inlier/outlier label (0,1)
            - src_keypts_corr:  [bs, num_corr, 3], the source points in the matched correspondences
            - tgt_keypts_corr:  [bs, num_corr, 3], the target points in the matched correspondences
        """
        self.gt_trans = gt_trans

        pred_tran = self.SC2_PCR(src_corr, tgt_corr)
        # final_tran = pred_tran

        final_tran, src_corr_final, tgt_corr_final = self.post_refinement_points(pred_tran, src_keypts, tgt_keypts, 20)
        # final_tran = self.post_refinement(pred_tran,  src_corr, tgt_corr, src_keypts, tgt_keypts, 20)

        frag1_warp = transform(src_corr, pred_tran)
        distance = torch.sum((frag1_warp - tgt_corr) ** 2, dim=-1) ** 0.5
        pred_labels = (distance < self.inlier_threshold)
        # src_corr_final = src_corr[:, pred_labels[0], :]
        # tgt_corr_final = tgt_corr[:, pred_labels[0], :]

        return final_tran, pred_labels, src_corr_final, tgt_corr_final


