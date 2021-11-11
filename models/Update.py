import torch, torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn.functional as F

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

    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.selected_clients = []
        self.datasetsplit = DatasetSplit(dataset, idxs)
        self.sub_dataloader = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.bsz, shuffle=True,
                                         collate_fn=self.my_collate_fn, drop_last=True)
        self.weight = args

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        epoch_loss = []
        for _ in range(self.args.local_epoch):
            batch_loss = []
            for i, batch in tqdm(enumerate(self.sub_dataloader),
                                 desc="Training",
                                 total=len(self.sub_dataloader)):
                text_feature = batch["query_tensor"].cuda(device=self.args.DEVICE_IDS[0])
                # bsz, max_len, 300
                visual_feature = batch['feature_tensor'].cuda(device=self.args.DEVICE_IDS[0])
                # bsz, max_fms, 1024
                fms_list = batch['fms_list']
                len_list = batch['len_list']

                score, index = net(visual_feature, text_feature, fms_list, len_list)
                gt_score = torch.zeros((self.args.bsz, self.args.max_fms)).cuda(self.args.DEVICE_IDS[0])
                gt_index = torch.zeros(self.args.bsz, 2).cuda(self.args.DEVICE_IDS[0])
                s_idx = (batch['clip_start_frame'] // self.args.segment_duration).cuda(self.args.DEVICE_IDS[0])
                e_idx = (batch['clip_end_frame'] // self.args.segment_duration).cuda(self.args.DEVICE_IDS[0])
                for idx in range(self.args.bsz):
                    gt_score[idx][int(s_idx[idx]):int(e_idx[idx])] = 1
                    gt_index[idx][0] = batch['clip_start_second'][idx] / fms_list[idx]
                    gt_index[idx][1] = batch['clip_end_second'][idx] / fms_list[idx]
                loss_sc = F.binary_cross_entropy(score.squeeze(-1), gt_score)
                loss_idx = F.mse_loss(index, gt_index)
                # loss_idx = F.binary_cross_entropy(pre_ts.squeeze(-1), gt_ts) + F.binary_cross_entropy(pre_te.squeeze(-1), gt_te)

                loss = loss_sc + self.args.lamda1 * loss_idx

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
        batch_tensor['feature_tensor'] = torch.empty(self.args.bsz, self.args.max_fms, 1024)
        batch_tensor['video_name'] = []
        batch_tensor['query_tensor'] = torch.empty(self.args.bsz, self.args.max_len, 300)
        batch_tensor['start_frame'] = torch.empty(self.args.bsz)
        batch_tensor['end_frame'] = torch.empty(self.args.bsz)
        batch_tensor['clip_start_frame'] = torch.empty(self.args.bsz)
        batch_tensor['clip_end_frame'] = torch.empty(self.args.bsz)
        batch_tensor["clip_start_second"] = torch.empty(self.args.bsz)
        batch_tensor["clip_end_second"] = torch.empty(self.args.bsz)
        for i, video in enumerate(batch):
            if video['feature_tensor'].shape[0] > self.args.max_fms:
                step = video['feature_tensor'].shape[0] // self.args.max_fms
                for new_fm, fm in enumerate(range(0, video['feature_tensor'].shape[0], step)):
                    if new_fm >= self.args.max_fms:
                        break
                    batch_tensor['feature_tensor'][i][new_fm] = video['feature_tensor'][fm]
            else:
                batch_tensor['feature_tensor'][i] = F.pad(video['feature_tensor'], (0, 0, 0, self.args.max_fms - fms_list[i]))
            batch_tensor['video_name'].append(video['video_name'])
            batch_tensor['query_tensor'][i] = video['query_tensor']
            batch_tensor['start_frame'][i] = video['start_frame']
            batch_tensor['end_frame'][i] = video['end_frame']
            batch_tensor['clip_start_frame'][i] = video['clip_start_frame']
            batch_tensor['clip_end_frame'][i] = video['clip_end_frame']
            batch_tensor['clip_start_second'][i] = video['clip_start_second']
            batch_tensor['clip_end_second'][i] = video['clip_end_second']

        if max(fms_list) != self.args.max_fms:
            fms_list[fms_list.index(max(fms_list))] = self.args.max_fms
        batch_tensor['fms_list'] = fms_list
        batch_tensor['len_list'] = len_list

        return batch_tensor