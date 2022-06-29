# from math import dist
import numpy as np
from MyDataloader import MyDataset
from distance import compute_distance_matrix
import torch
from torch.autograd import Variable
import tqdm


def extract_info(model, dataloaders, feat_len=2048):
    features = torch.FloatTensor()
    labels = []
    pic_names = []
    camids = []
    count = 0
    for data in dataloaders:
        imgs, camid, label, pic_name = data
        camids.extend(camid.tolist())
        labels.extend(label.tolist())
        pic_names.extend(pic_name)
        n, c, h, w = imgs.size()
        count += n
        ff = torch.FloatTensor(n, feat_len).zero_().cuda()

        input_img = Variable(imgs.cuda())  # input_img = imgs
        with torch.no_grad():
            outputs = model(input_img)

        # print(outputs.size())
        ff += outputs
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        # print(ff.shape)
        features = torch.cat((features, ff.data.cpu().float()), 0)  # catenate
    return features, camids, labels, pic_names


'''
    return CMC and mAP
'''


def calculate_result(gallery_feature, gallery_label, query_feature, query_label,
                     result_file="./result.txt", selftest=True, check_range=10,
                     draw=False):
    # calculate distance matrix
    query_feature = torch.FloatTensor(query_feature).cuda()
    gallery_feature = torch.FloatTensor(gallery_feature).cuda()
    dist_mat = compute_distance_matrix(query_feature, gallery_feature)
    dist_mat = dist_mat.cpu().numpy()  # TODO: check whether narray

    if selftest:
        # exclude themselves
        np.fill_diagonal(dist_mat, 1e10)

    rank = np.argsort(dist_mat, axis=1)

    cmc = []
    cmc_last = [0 for i in range(len(gallery_label))]
    for idx in range(check_range):
        cmc_idx = [gallery_label[rank[i][idx]] == gallery_label[i]
                   for i in range(len(gallery_label))]  # TODO:check it
        cmc_last = np.logical_or(cmc_last, cmc_idx)
        cmc_tmp = 1.0*sum(cmc_last)/len(gallery_label)
        cmc.append(cmc_tmp)

    ap = 0
    for q, ordered_idx in enumerate(rank):
        ordered_idx = ordered_idx[:-1].astype(int)  # remove themselves
        ordered_label = np.array(gallery_label)[ordered_idx]
        pos = np.where(ordered_label == gallery_label[q])[0]
        length = pos.shape[0]  # check the shape

        for i, p in enumerate(pos):
            ap += 1.0/length*(i+1)/(p+1)
    mAP = ap/len(gallery_label)

    str_result = 'Result:\nRank@1:%f\nRank@5:%f\nRank@10:%f\nmAP:%f\n' % (
        cmc[0], cmc[4], cmc[9], mAP)
    # str_result = 'Result: \nRank@1:%f \nRank@5:%f\n  mAP:%f\n' % (
    #     cmc[0], cmc[4], mAP)
    print(str_result)
    return cmc, mAP


if __name__ == "__main__":
    import sys
    sys.path.append("/home/Mine")
    model = []
    gf = np.arange(9.).reshape(3, 3)
    gf = np.tile(gf, (3, 1))
    gf[1] += 0.1
    gf[2] += 0.3
    gl = np.arange(3)
    gl = np.tile(gl, 3)
    gl[3], gl[4] = gl[4], gl[3]
    gl[6], gl[8] = gl[8], gl[6]
    calculate_result(gf, gl, gf, gl, check_range=5)
