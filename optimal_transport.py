import torch
import numpy as np
'''
Reference:
https://github.com/magicleap/SuperGluePretrainedNetwork/blob/c0626d58c843ee0464b0fa1dd4de4059bfae0ab4/models/superglue.py#L150
'''


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    '''
    Perform Sinkhorn Normalization in Log-space for stability
    :param Z:
    :param log_mu:
    :param log_nu:
    :param iters:
    :return:
    '''
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)

    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, bins0=None, bins1=None, alpha=None, iters=100):
    '''
    Perform Differentiable Optimal Transport in Log-space for stability
    :param scores:
    :param alpha:
    :param iters:
    :return:
    '''

    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    if bins0 is None:
        bins0 = alpha.expand(b, m, 1)
    if bins1 is None:
        bins1 = alpha.expand(b, 1, n)

    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm #multiply probabilities by M + N
    return Z


def rpmnet_sinkhorn(log_score, bins0, bins1, iters: int):
    b, m, n = log_score.shape
    alpha = torch.zeros(size=(b, 1, 1)).cuda()
    log_score_padded = torch.cat([torch.cat([log_score, bins0], -1),
                                  torch.cat([bins1, alpha], -1)], 1)

    for i in range(iters):
        #Row Normalization
        log_score_padded = torch.cat((
            log_score_padded[:, :-1, :] - (torch.logsumexp(log_score_padded[:, :-1, :], dim=2, keepdim=True)),
            log_score_padded[:, -1, None, :]),
            dim=1)

        #Column Normalization
        log_score_padded = torch.cat((
            log_score_padded[:, :, :-1] - (torch.logsumexp(log_score_padded[:, :, :-1], dim=1, keepdim=True)),
            log_score_padded[:, :, -1, None]),
            dim=2)



    return log_score_padded


def correspondences_from_score_max(score, mutual=False, supp=False, certainty=None, return_score=False, thres=None):
    '''
    Return estimated rough matching regions from score matrix
    param: score: score matrix, [N, M]
    return: correspondences [K, 2]
    '''
    score = torch.exp(score)
    row_idx = torch.argmax(score[:-1, :], dim=1)
    # print(row_idx.shape)
    # src_local_corr = score.gather(dim=1, index=corr[:, :, 0].unsqueeze(-1).expand(-1, -1, 3))
    row_seq = torch.arange(row_idx.shape[0]).cuda()

    col_idx = torch.argmax(score[:, :-1], dim=0)
    col_seq = torch.arange(col_idx.shape[0]).cuda()


    row_map = torch.zeros_like(score).cuda().bool()
    row_map[row_seq, row_idx] = True
    col_map = torch.zeros_like(score).cuda().bool()
    col_map[col_idx, col_seq] = True

    if mutual:
        sel_map = torch.logical_and(row_map, col_map)[:-1, :-1]
    else:
        sel_map = torch.logical_or(row_map, col_map)[:-1, :-1]

    if thres is not None:
        add_map = (score[:-1, :-1] >= thres)
        sel_map = torch.logical_and(sel_map, add_map)

    correspondences = sel_map.nonzero(as_tuple=False)

    if supp and correspondences.shape[0] == 0:
        correspondences = torch.zeros(1, 2).long().cuda()
    if return_score:
        corr_score = score[correspondences[:, 0], correspondences[:, 1]]
        return correspondences, corr_score.view(-1)
    else:
        return correspondences

def get_fine_grained_correspondences(scores, mutual=False, supp=False, certainty=None, node_corr_conf=None, thres=None):
    '''
    '''
    b, n, m = scores.shape[0], scores.shape[1] - 1, scores.shape[2] - 1
    src_idx_base = 0
    tgt_idx_base = 0

    correspondences = torch.empty(size=(0, 2), dtype=torch.int32).cuda()
    corr_fine_score = torch.empty(size=(0, 1), dtype=torch.float32).cuda()

    for i in range(b):
        score = scores[i, :, :]
        if node_corr_conf is not None:
            correspondence, fine_score = correspondences_from_score_max(score, mutual=mutual, supp=supp, certainty=certainty, return_score=True, thres=thres)
            fine_score = fine_score * node_corr_conf[i]
        else:
            correspondence = correspondences_from_score_max(score, mutual=mutual, supp=supp, certainty=certainty, return_score=False, thres=thres)


        correspondence[:, 0] += src_idx_base
        correspondence[:, 1] += tgt_idx_base

        correspondences = torch.cat([correspondences, correspondence], dim=0)
        if node_corr_conf is not None:
            corr_fine_score = torch.cat([corr_fine_score, fine_score.unsqueeze(-1)], dim=0)

        src_idx_base += n
        tgt_idx_base += m
    if node_corr_conf is not None:
        return correspondences, corr_fine_score
    else:
        return correspondences


def get_fine_correspondences(scores, src_corr_knn_idx, tgt_corr_knn_idx, mutual=False, supp=False, certainty=None, node_corr_conf=None, thres=None, use_sampling=True):

    b, n, m = scores.shape[0], scores.shape[1] - 1, scores.shape[2] - 1

    correspondences = torch.empty(size=(0, 2), dtype=torch.int32).cuda()
    corr_fine_score = torch.empty(size=(0, 1), dtype=torch.float32).cuda()

    total_local_corr_idx = torch.empty(size=(0, 2), dtype=torch.int32).cuda()

    for i in range(b):
        score = scores[i, :, :]
        if node_corr_conf is not None:
            correspondence, fine_score = correspondences_from_score_max(score, mutual=mutual, supp=supp,
                                                                        certainty=certainty, return_score=True,
                                                                        thres=thres)
            fine_score = fine_score * node_corr_conf[i]
        else:
            correspondence, fine_score = correspondences_from_score_max(score, mutual=mutual, supp=supp, certainty=certainty,
                                                            return_score=True, thres=thres)
            if i ==0:
                total_fine_score = fine_score
            else:
                total_fine_score = torch.cat([total_fine_score, fine_score], dim=0)
            # src_corr_knn_idx [bs, num_seeds,knn_num]
            src_local_corr_idx = src_corr_knn_idx[:, i, :].gather(dim=1, index=correspondence[None, :, 0])
            tgt_local_corr_idx = tgt_corr_knn_idx[:, i, :].gather(dim=1, index=correspondence[None, :, 1])
            local_corr_idx = torch.cat([src_local_corr_idx.unsqueeze(-1), tgt_local_corr_idx.unsqueeze(-1)], dim=-1).view(-1, 2)
            total_local_corr_idx = torch.cat([total_local_corr_idx, local_corr_idx], dim=0)


    prob = total_fine_score / total_fine_score.sum()
    if total_fine_score.shape[0]>500 and use_sampling:
        idx = np.arange(total_fine_score.shape[0])
        idx = np.random.choice(idx, size=500, replace=False, p=prob.cpu().numpy())
        total_local_corr_idx = total_local_corr_idx[torch.from_numpy(idx).cuda()]

    local_corr_idx_unique = torch.unique(total_local_corr_idx, dim=0)
    # total_fine_score = total_fine_score[local_corr_idx_unique_idx]
    # total_fine_score_unique = torch.unique(total_fine_score, dim=0, sorted=False)

    return local_corr_idx_unique

def get_correspondences(scores, src_corr_knn_idx, tgt_corr_knn_idx, mutual=False, supp=False, certainty=None, node_corr_conf=None, thres=None, use_sampling=True):

    b, n, m = scores.shape[0], scores.shape[1] - 1, scores.shape[2] - 1
    # src_corr_knn_idx [bs, num_seeds, knn_num]
    corr, corr_score = correspondences_from_score_max_1(scores, return_score=True)
    # corr [num_seeds, knn_num*2, 2], corr_score
    src_local_corr_idx = src_corr_knn_idx.gather(dim=2, index=corr[None, :, :, 0])
    tgt_local_corr_idx = tgt_corr_knn_idx.gather(dim=2, index=corr[None, :, :, 1])  # [bs, num_seeds, knn_num*2]
    local_corr_idx = torch.cat([src_local_corr_idx.unsqueeze(-1), tgt_local_corr_idx.unsqueeze(-1)], dim=-1).view(-1, 2)
    corr_score = corr_score.view(-1)

    prob = corr_score / corr_score.sum()
    if corr_score.shape[0]>500 and use_sampling:
        idx = np.arange(corr_score.shape[0])
        idx = np.random.choice(idx, size=500, replace=False, p=prob.cpu().numpy())
        local_corr_idx = local_corr_idx[torch.from_numpy(idx).cuda()]

    sampling_idx_unique = torch.unique(local_corr_idx, dim=0)
    # print("对应关系总数", sampling_idx_unique.shape[0])

    return sampling_idx_unique

def get_correspondences2(scores, src_corr_knn_idx, tgt_corr_knn_idx):

    b, n, m = scores.shape[0], scores.shape[1] - 1, scores.shape[2] - 1
    # src_corr_knn_idx [bs, num_seeds, knn_num]
    corr, not_nan_mask = correspondences_from_score_max_1(scores, return_score=False)
    # corr [num_seeds, knn_num*2, 2], corr_score
    src_local_corr_idx = src_corr_knn_idx.gather(dim=2, index=corr[None, :, :, 0])
    tgt_local_corr_idx = tgt_corr_knn_idx.gather(dim=2, index=corr[None, :, :, 1])  # [bs, num_seeds, knn_num*2]
    local_corr_idx = torch.cat([src_local_corr_idx.unsqueeze(-1), tgt_local_corr_idx.unsqueeze(-1)], dim=-1)

    return local_corr_idx[0], not_nan_mask


def correspondences_from_score_max_2(score, return_score=False):
    '''
    Return estimated rough matching regions from score matrix
    param: score: score matrix, [bs, N, M]
    return: correspondences [bs, K, 2]
    '''
    score = torch.exp(score)[:, :-1, :-1]
    num_seeds, num_knn, num_knn1 = score.shape
    row_idx = torch.argmax(score, dim=2)
    corr_row = torch.cat([torch.arange(num_knn1)[None, :, None].expand(num_seeds, -1, -1).cuda(), row_idx[:, :, None]], dim=-1)
    col_idx = torch.argmax(score, dim=1)
    corr_col = torch.cat([col_idx[:, :, None], torch.arange(num_knn)[None, :, None].expand(num_seeds, -1, -1).cuda()], dim=-1)
    corr = torch.cat([corr_row, corr_col], dim=1)


    # distance = torch.norm((src_feature_knn[:, :, None, :] - tgt_feature_knn[:, None, :, :]), dim=-1)
    # # [num_seeds,knn_num,knn_num]
    # # match points in feature space. # [num_seeds,knn_num,knn_num]
    # source_idx = torch.argmin(distance, dim=2)
    # corr = torch.cat([torch.arange(num_knn)[None, :, None].expand(num_seeds,-1,-1).cuda(), source_idx[:, :, None]], dim=-1)

    if return_score:
        # batch_indices = torch.arange(num_seeds).unsqueeze(1).expand(-1, num_knn*2)  # 创建一个形状为(k, 1)的张量，用于扩展到(k, m)
        # row_indices_expanded = corr[:, :, 0].unsqueeze(2)  # 将行索引扩展为(k, m, 1)
        # col_indices_expanded = corr[:, :, 1].unsqueeze(2)  # 将列索引扩展为(k, m, 1)
        # corr_score = score[batch_indices, row_indices_expanded, col_indices_expanded]
        corr_score = score[torch.arange(num_seeds).unsqueeze(1), corr[:, :, 1], corr[:, :, 0]]
        return corr, corr_score
    else:
        return corr, None

# nan mask
def correspondences_from_score_max_1(score, return_score=False):
    '''
    Return estimated rough matching regions from score matrix
    param: score: score matrix, [bs, N, M]
    return: correspondences [bs, K, 2]
    '''
    score = torch.exp(score)
    num_seeds = score.shape[0]


    row_idx = torch.argmax(score[:, :-1, :], dim=2)
    not_nan_mask_row = row_idx != score.shape[1]-1
    # ****超出范围的索引先用0代替****


    row_idx[~not_nan_mask_row]=0
    corr_row = torch.cat([torch.arange(row_idx.shape[1])[None, :, None].expand(num_seeds, -1, -1).cuda(), row_idx[:, :, None]], dim=-1)

    col_idx = torch.argmax(score[:, :, :-1], dim=1)
    not_nan_mask_col = col_idx != score.shape[2]-1
    # ****超出范围的索引先用0代替****

    col_idx[~not_nan_mask_col]=0
    corr_col = torch.cat([col_idx[:, :, None], torch.arange(col_idx.shape[1])[None, :, None].expand(num_seeds, -1, -1).cuda()], dim=-1)

    corr = torch.cat([corr_row, corr_col], dim=1)
    not_nan_mask = torch.cat([not_nan_mask_row, not_nan_mask_col], dim=1).unsqueeze(2).expand(-1, -1, 2)

    # torch.unique(local_corr_idx, dim=0)

    # distance = torch.norm((src_feature_knn[:, :, None, :] - tgt_feature_knn[:, None, :, :]), dim=-1)
    # # [num_seeds,knn_num,knn_num]
    # # match points in feature space. # [num_seeds,knn_num,knn_num]
    # source_idx = torch.argmin(distance, dim=2)
    # corr = torch.cat([torch.arange(num_knn)[None, :, None].expand(num_seeds,-1,-1).cuda(), source_idx[:, :, None]], dim=-1)

    if return_score:
        # batch_indices = torch.arange(num_seeds).unsqueeze(1).expand(-1, num_knn*2)  # 创建一个形状为(k, 1)的张量，用于扩展到(k, m)
        # row_indices_expanded = corr[:, :, 0].unsqueeze(2)  # 将行索引扩展为(k, m, 1)
        # col_indices_expanded = corr[:, :, 1].unsqueeze(2)  # 将列索引扩展为(k, m, 1)
        # corr_score = score[batch_indices, row_indices_expanded, col_indices_expanded]
        corr_score = score[torch.arange(num_seeds).unsqueeze(1), corr[:, :, 1], corr[:, :, 0]]
        return corr, corr_score, not_nan_mask
    else:
        return corr, not_nan_mask
