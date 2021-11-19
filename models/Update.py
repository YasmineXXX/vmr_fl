import torch, torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn.functional as F

def get_iou(groundtruth, predict):
    groundtruth_init = max(0,groundtruth[0])
    groundtruth_end = groundtruth[1]
    predict_init = max(0,predict[0])
    predict_end = predict[1]
    init_min = min(groundtruth_init,predict_init)
    end_max = max(groundtruth_end,predict_end)
    init_max = max(groundtruth_init,predict_init)
    end_min = min(groundtruth_end,predict_end)
    if end_min < init_max:
        return 0
    IOU = ( end_min - init_max ) * 1.0 / ( end_max - init_min)
    return IOU

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        dataset = self.dataset[self.idxs[item]]
        return dataset


class LocalUpdate(object):

    def __init__(self, config, number, dataset=None, idxs=None):
        self.config = config
        self.number = number
        self.selected_clients = []
        # self.datasetsplit = DatasetSplit(dataset, idxs)
        # self.sub_dataloader = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.config.bsz, shuffle=True,
        #                                  collate_fn=self.my_collate_fn, drop_last=True)
        self.sub_dataloader = DataLoader(dataset, batch_size=self.config.bsz, shuffle=True,
                                          collate_fn=self.my_collate_fn, drop_last=True, num_workers=4)
        self.weight = config

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        epoch_loss = []
        des = 'user {:d} '.format(self.number)
        for _ in range(self.config.local_epoch):
            batch_loss = []
            for i, batch in tqdm(enumerate(self.sub_dataloader),
                                 desc=des+"Training",
                                 total=len(self.sub_dataloader)):
                text_feature = batch["query_tensor"].cuda(device=self.config.DEVICE_IDS[0])
                # bsz, max_len, 300
                visual_feature = batch['feature_tensor'].cuda(device=self.config.DEVICE_IDS[0])
                # bsz, max_fms, 1024
                fms_list = batch['fms_list']
                len_list = batch['len_list']

                score, index = net(visual_feature, text_feature, fms_list, len_list)
                gt_score = batch['gt_score'].cuda(device=self.config.DEVICE_IDS[0])
                gt_index = batch['gt_index'].cuda(device=self.config.DEVICE_IDS[0])
                weight_balance = 1e4
                loss1 = F.binary_cross_entropy(score, gt_score, reduction='none').sum(dim=-1)
                loss2 = torch.nn.MSELoss(reduction='none')(index, gt_index).sum(-1)
                masks = torch.zeros(len(index)).float().cuda(self.config.DEVICE_IDS[0])
                for i in range(len(masks)):
                    iou = get_iou(index[i], gt_index[i])
                    masks[i] = (1 - iou)
                loss1 = loss1 * masks
                loss2 = loss2 * masks
                loss = torch.sum(loss1 * weight_balance + loss2) / len(gt_score)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def my_collate_fn(self, batch):
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
        bsz = len(batch)

        fms_list = [e['fms'] for e in batch]
        len_list = [e['len'] for e in batch]

        batch_tensor = {}
        batch_tensor['feature_tensor'] = torch.empty(self.config.bsz, self.config.max_fms, self.config.vdim)
        batch_tensor['video_name'] = []
        batch_tensor['query_tensor'] = torch.empty(self.config.bsz, self.config.max_len, self.config.sdim)
        batch_tensor["clip_start_second"] = torch.empty(self.config.bsz)
        batch_tensor["clip_end_second"] = torch.empty(self.config.bsz)
        batch_tensor["duration"] = torch.empty(self.config.bsz)
        batch_tensor["gt_score"] = torch.empty(self.config.bsz, self.config.max_fms)
        batch_tensor["gt_index"] = torch.empty(self.config.bsz, 2)
        for i, video in enumerate(batch):
            batch_tensor['feature_tensor'][i] = torch.tensor(video['feature_tensor'])
            batch_tensor['video_name'].append(video['video_name'])
            batch_tensor['query_tensor'][i] = video['query_tensor']
            batch_tensor['clip_start_second'][i] = video['clip_start_second']
            batch_tensor['clip_end_second'][i] = video['clip_end_second']
            batch_tensor["duration"][i] = video["duration"]
            batch_tensor["gt_score"][i] = video["gt_score"]
            batch_tensor["gt_index"][i] = video["gt_index"]

        if max(fms_list) != self.config.max_fms:
            fms_list[fms_list.index(max(fms_list))] = self.config.max_fms
        batch_tensor['fms_list'] = fms_list
        batch_tensor['len_list'] = len_list

        return batch_tensor