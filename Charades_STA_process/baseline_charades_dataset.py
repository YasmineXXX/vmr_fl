import numpy as np
import torch
import h5py
import nltk
from skimage.transform import resize

class VideoRecord(object):
    def __init__(self, row, root_datapath):
        # row for items in annotation file, root_datapath for feature dir of root path
        self._data = row
        # AO8RW_1_809_a person is putting a book on a shelf._0_165_0.0_6.9
        # self._path = os.path.join(root_datapath, row[0]) + ".npy"
        self._path = root_datapath
        # /data/wangyan/dataset/data/Charades/charades_i3d_rgb.hdf5


    @property
    def path(self):
        return self._path

    @property
    def query_sent(self):
        return self._data[1]

    @property
    def clip_start_second(self):
        return float(self._data[2])

    @property
    def clip_end_second(self):
        return float(self._data[3])

    @property
    def duration(self):
        return float(self._data[4])

    @property
    def label(self):
        return self._data[0]

class VideoFrameDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_path: str,
                 annotationfile_path: str,
                 token_path: str,
                 max_fms: int,
                 max_len: int):
        super(VideoFrameDataset, self).__init__()

        self.root_path = root_path
        # "/data/wangyan/dataset/data/activity/sub_activitynet_v1-3.c3d.hdf5"
        self.annotationfile_path = annotationfile_path
        # "./activity_annotations.txt"
        self.glove = np.load(token_path, allow_pickle=True).tolist()
        self.max_fms = max_fms
        self.max_len = max_len

        self._parse_list()

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(">"), self.root_path) for x in open(self.annotationfile_path)]

    def __getitem__(self, index):
        """
        Returns:
            a dict which includes:
            1) feature_tensor: a list of PIL images or the result
            of applying self.transform on this list if
            self.transform is not None.
            2) query_tensor: query sentence
            3) clip_start_second:
            4) clip_end_second:
            5) video_name:
            6) duration:
            7) fms
            8) len
        """
        record = self.video_list[index]

        sample = {}
        sample["feature_tensor"], sample["fms"] = self._get(record)
        sample["query_tensor"], sample["len"]= self.text2feature(record.query_sent)
        sample["clip_start_second"] = record.clip_start_second
        sample["clip_end_second"] = record.clip_end_second
        sample["video_name"] = record.label
        sample["duration"] = record.duration

        s_idx = int(sample['clip_start_second'] / sample['duration'] * self.max_fms)
        e_idx = int(sample['clip_end_second'] / sample['duration'] * self.max_fms)
        sample['gt_score'] = torch.zeros(self.max_fms)
        sample['gt_score'][s_idx:e_idx] = 1
        sample['gt_index'] = torch.zeros(2)
        sample['gt_index'][0] = s_idx
        sample['gt_index'][1] = e_idx
        return sample

    def _get(self, record):

        i3d_feature = h5py.File(record.path, 'r')
        vid = record.label
        # feature = torch.tensor(i3d_feature[vid]['i3d_rgb_features'])
        v_dim = 1024
        feature = resize(i3d_feature[vid]['i3d_rgb_features'], (self.max_fms, v_dim))

        return feature, self.max_fms
        # return feature, min(len(feature), self.max_fms)


    def text2feature(self, sentence):
        s_feature = np.zeros([self.max_len, 300], dtype=np.float32)
        words = nltk.word_tokenize(sentence.lower())
        for i, w in enumerate(words):
            if i >= self.max_len:
                break
            try:
                f = self.glove[w]
            except KeyError:
                f = np.random.randn(300, )
            s_feature[i] = f
        return torch.tensor(s_feature), min(len(words), self.max_len)

    def __len__(self):
        return len(self.video_list)
