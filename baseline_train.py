import os
import torch, torchvision
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import argparse, json, sys, time
from tqdm import tqdm
from easydict import EasyDict as edict
from torch.utils.data.dataloader import default_collate
from Charades_STA_process import baseline_charades_dataset
from Baseline import Baseline

def get_pretraining_args():
    desc = "shared config for pretraining and finetuning"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--feature_dir", type=str, help="eg: /data/wangyan/dataset/data/Charades/charades_i3d_rgb.hdf5",
                        default="/data/wangyan/dataset/data/Charades/charades_i3d_rgb.hdf5")
    parser.add_argument("--annotation_file", type=str, help="eg: ./Charades_STA_process/charades_annotations.txt",
                        default="./Charades_STA_process/charades_annotations.txt")
    parser.add_argument("--val_annotation_file", type=str, help="eg: ./Charades_STA_process/charades_annotations_test.txt",
                        default="./Charades_STA_process/charades_annotations_test.txt")
    parser.add_argument("--token", type=str, help="eg: /data/wangyan/Charades-STA/6B.300d.npy",
                        default="/data/wangyan/Charades-STA/6B.300d.npy")
    parser.add_argument("--result_dir", type=str, default="./result_baseline",
        help="dir to store model checkpoints & training meta.")
    parser.add_argument("--best_train_model_file_name", type=str, default="/best_model.ckpt",
        help="best train model file name, don't forget a '/' ahead.")
    parser.add_argument("--train_model_file_name", type=str, default="/model.ckpt",
        help="train model file name, don't forget a '/' ahead.")
    parser.add_argument("--config", default="./hyper_param.json", help="JSON config files")
    parsed_args = parser.parse_args()
    args = edict(vars(parsed_args))
    config_args = json.load(open(args.config))
    override_keys = {arg[2:].split("=")[0] for arg in sys.argv[1:]
                     if arg.startswith("--")}
    for k, v in config_args.items():
        if k not in override_keys:
            setattr(args, k, v)
    return args

config = get_pretraining_args()

def my_collate_fn(batch):
    # input: a list of bsz dicts,
    # the structure of each one(batch) is:
    # 'feature_tensor': a tensor:
    #        [fms, C, H, W]
    # 'query_sent': ['a person is putting a book on a shelf.']
    # 'start_frame': tensor([1])
    # 'end_frame': tensor([809])
    # 'clip_start_frame': tensor([0])
    # 'clip_end_frame': tensor([165])
    # 'video_name': ('AO8RW',)

    fms_list = [e['fms'] for e in batch]
    len_list = [e['len'] for e in batch]

    batch_tensor = {}
    batch_tensor['feature_tensor'] = torch.empty(config.bsz, config.max_fms, 1024)
    batch_tensor['video_name'] = []
    batch_tensor['query_tensor'] = torch.empty(config.bsz, config.max_len, 300)
    batch_tensor['start_frame'] = torch.empty(config.bsz)
    batch_tensor['end_frame'] = torch.empty(config.bsz)
    batch_tensor['clip_start_frame'] = torch.empty(config.bsz)
    batch_tensor['clip_end_frame'] = torch.empty(config.bsz)
    batch_tensor["clip_start_second"] = torch.empty(config.bsz)
    batch_tensor["clip_end_second"] = torch.empty(config.bsz)
    for i, video in enumerate(batch):
        if video['feature_tensor'].shape[0] > config.max_fms:
            step = video['feature_tensor'].shape[0] // config.max_fms
            for new_fm, fm in enumerate(range(0, video['feature_tensor'].shape[0], step)):
                if new_fm >= config.max_fms:
                    break
                batch_tensor['feature_tensor'][i][new_fm] = video['feature_tensor'][fm]
        else:
            batch_tensor['feature_tensor'][i] = F.pad(video['feature_tensor'], (0, 0, 0, config.max_fms-fms_list[i]))
        batch_tensor['video_name'].append(video['video_name'])
        batch_tensor['query_tensor'][i] = video['query_tensor']
        batch_tensor['start_frame'][i] = video['start_frame']
        batch_tensor['end_frame'][i] = video['end_frame']
        batch_tensor['clip_start_frame'][i] = video['clip_start_frame']
        batch_tensor['clip_end_frame'][i] = video['clip_end_frame']
        batch_tensor['clip_start_second'][i] = video['clip_start_second']
        batch_tensor['clip_end_second'][i] = video['clip_end_second']

    if max(fms_list) != config.max_fms:
        fms_list[fms_list.index(max(fms_list))] = config.max_fms
    batch_tensor['fms_list'] = fms_list
    batch_tensor['len_list'] = len_list

    return batch_tensor

def compute_acuracy(pred_start_second, pred_end_second,
                    gt_start_second, gt_end_second, iou_thds=(0.1, 0.3, 0.5, 0.7)):
    # pred_start_frame: bsz
    # gt_start_frame: bsz
    intersection = np.maximum(0, np.minimum(pred_end_second, gt_end_second) - np.maximum(pred_start_second, gt_start_second))
    union = np.maximum(pred_end_second, gt_end_second) - np.minimum(pred_start_second, gt_start_second)  # not the correct union though
    # iou_scores = float(intersection) / (union + 1e-6)
    iou_scores = np.divide(intersection, (union + 1e-6), out=np.zeros_like(intersection), where=union != 0)
    iou_corrects_batch = {}
    for iou_thd in iou_thds:
        iou_corrects_batch[str(iou_thd)] = 0
    for iou_thd in iou_thds:
        iou_corrects_batch[str(iou_thd)] += (sum(iou_scores >= iou_thd))
    return iou_corrects_batch

def get_iou(pred, gt):
    intersection = max(0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])
    return float(intersection) / (union + 1e-6)

def start_validation(val_loader, model, train_log_filepath, epoch):
    iou_value = []

    acc_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Accuracy] {acc:s}\n"
    for i, batch in tqdm(enumerate(val_loader),
                         desc="Validation",
                         total=len(val_loader)):
        text_feature = batch["query_tensor"].cuda(device=config.DEVICE_IDS[0])
        # bsz, max_len, 300
        visual_feature = batch['feature_tensor'].cuda(device=config.DEVICE_IDS[0])
        # bsz, max_fms, 1024
        fms_list = batch['fms_list']
        len_list = batch['len_list']

        score, index = model(visual_feature, text_feature, fms_list, len_list)
        # pre_index = torch.zeros(config.bsz, 2).cuda(config.DEVICE_IDS[0])
        # for idx in range(config.bsz):
        #     pre_index[idx][0] = index[idx][0] * fms_list[idx]
        #     pre_index[idx][1] = index[idx][1] * fms_list[idx]
        gt_index = torch.zeros(config.bsz, 2).cuda(config.DEVICE_IDS[0])
        for idx in range(config.bsz):
            gt_index[idx][0] = batch['clip_start_second'][idx] / fms_list[idx]
            gt_index[idx][1] = batch['clip_end_second'][idx] / fms_list[idx]

        index = index.detach().cpu().numpy()
        gt_index = gt_index.cpu().numpy()

        for i in range(config.bsz):
            iou = get_iou(gt_index[i], index[i])
            iou_value.append(iou)

    ious = np.array(iou_value)
    iou1 = np.average(ious > 0.1)
    iou3 = np.average(ious > 0.3)
    iou5 = np.average(ious > 0.5)
    iou7 = np.average(ious > 0.7)
    iou9 = np.average(ious > 0.9)
    acc = "iou1:" + str(iou1) + ", iou3:" + str(iou3) + ", iou5:" + str(iou5) + ", iou7:" + str(iou7) + ", iou9:" + str(iou9)

    to_write_acc = acc_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H:%M:%S"),
                                                epoch=epoch,
                                                acc=acc)
    with open(train_log_filepath, "a") as f:
        f.write(to_write_acc)

def start_training():
    root = os.path.join(config.feature_dir)
    annotation_file = os.path.join(config.annotation_file)
    token = os.path.join(config.token)

    dataset = baseline_charades_dataset.VideoFrameDataset(
        root_path=root,
        annotationfile_path=annotation_file,
        token_path=token,
        max_fms=config.max_fms,
        max_len=config.max_len,
    )

    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=config.bsz,
                                               shuffle=True, collate_fn=my_collate_fn, drop_last=True)

    val_annotation_file = os.path.join(config.val_annotation_file)
    val_dataset = baseline_charades_dataset.VideoFrameDataset(
        root_path=root,
        annotationfile_path=val_annotation_file,
        token_path=token,
        max_fms=config.max_fms,
        max_len=config.max_len,
    )

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=config.bsz,
                                               shuffle=True, collate_fn=my_collate_fn, drop_last=True)

    result_dir = config.result_dir

    model = Baseline(config)
    # model.to(device)
    model = torch.nn.DataParallel(model, device_ids=config.DEVICE_IDS)  # 声明所有可用设备
    model = model.cuda(device=config.DEVICE_IDS[0])  # 模型放在主设备
    model.requires_grad_(True)

    # optimizer = optim.SGD(params=model.parameters(), lr=config.learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    train_log_filename = "train_log_100_epoch.txt"
    train_log_filepath = os.path.join(result_dir, train_log_filename)
    train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"

    min_loss = 10000

    for epoch in range(config.num_epochs):

        epoch_mean_loss = 0
        for i, batch in tqdm(enumerate(train_loader),
                             desc="Training",
                             total=len(train_loader)):
            text_feature = batch["query_tensor"].cuda(device=config.DEVICE_IDS[0])
            # bsz, max_len, 300
            visual_feature = batch['feature_tensor'].cuda(device=config.DEVICE_IDS[0])
            # bsz, max_fms, 1024
            fms_list = batch['fms_list']
            len_list = batch['len_list']

            score, index = model(visual_feature, text_feature, fms_list, len_list)
            gt_score = torch.zeros((config.bsz, config.max_fms)).cuda(config.DEVICE_IDS[0])
            gt_index = torch.zeros(config.bsz, 2).cuda(config.DEVICE_IDS[0])
            s_idx = (batch['clip_start_frame']//config.segment_duration).cuda(config.DEVICE_IDS[0])
            e_idx = (batch['clip_end_frame']//config.segment_duration).cuda(config.DEVICE_IDS[0])
            for idx in range(config.bsz):
                gt_score[idx][int(s_idx[idx]):int(e_idx[idx])] = 1
                gt_index[idx][0] = batch['clip_start_second'][idx] / fms_list[idx]
                gt_index[idx][1] = batch['clip_end_second'][idx] / fms_list[idx]
            loss_sc = F.binary_cross_entropy(score.squeeze(-1), gt_score)
            loss_idx = F.mse_loss(index, gt_index)
            # loss_idx = F.binary_cross_entropy(pre_ts.squeeze(-1), gt_ts) + F.binary_cross_entropy(pre_te.squeeze(-1), gt_te)

            loss = loss_sc + config.lamda1 * loss_idx

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_mean_loss += loss

        epoch_mean_loss /= len(train_loader)
        print("\nloss:", float(epoch_mean_loss))
        if epoch_mean_loss < min_loss:
            min_loss = epoch_mean_loss
            checkpoint = {
                "model_param": model.state_dict(),
                "model_cfg": config}
            torch.save(checkpoint, result_dir + config.best_train_model_file_name)

        start_validation(val_loader, model, train_log_filepath, epoch)

        to_write = train_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H:%M:%S"),
                                                  epoch=epoch,
                                                  loss_str=" ".join(["{}".format(epoch_mean_loss)]))
        with open(train_log_filepath, "a") as f:
            f.write(to_write)

    checkpoint = {
        "model_struct": model,
        "model_param": model.state_dict(),
        "model_cfg": config}
    torch.save(checkpoint, result_dir + config.train_model_file_name)

if __name__ == '__main__':
    start_training()