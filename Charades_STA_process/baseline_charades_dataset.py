import numpy as np
import torch
import h5py
import nltk

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
    def num_frames(self):
        return self.end_frame - self.start_frame + 1  # +1 because end frame is inclusive

    @property
    def start_frame(self):
        return int(self._data[1])

    @property
    def end_frame(self):
        return int(self._data[2])

    @property
    def query_sent(self):
        return self._data[3]

    @property
    def clip_start_frame(self):
        return int(self._data[4])

    @property
    def clip_end_frame(self):
        return int(self._data[5])

    @property
    def clip_start_second(self):
        return float(self._data[6])

    @property
    def clip_end_second(self):
        return float(self._data[7])

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
        # "/data/wangyan/dataset/data/Charades/charades_i3d_rgb.hdf5"
        self.annotationfile_path = annotationfile_path
        # "./Charades_STA_process/charades_annotations.txt"
        self.glove = np.load(token_path, allow_pickle=True).tolist()
        self.max_fms = max_fms
        self.max_len = max_len

        self._parse_list()

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split("_"), self.root_path) for x in open(self.annotationfile_path)]

    def __getitem__(self, index):
        """
        Returns:
            a dict which includes:
            1) feature_tensor: a list of PIL images or the result
            of applying self.transform on this list if
            self.transform is not None.
            2) query_tensor: query sentence
            3) start_frame:
            4) end_frame:
            5) clip_start_frame:
            6) clip_end_frame:
            7) clip_start_second:
            8) clip_end_second:
            9) video_name:
            10) fms
            11) len
        """
        record = self.video_list[index]

        sample = {}
        sample["feature_tensor"], sample["fms"] = self._get(record)
        sample["query_tensor"], sample["len"]= self.text2feature(record.query_sent)
        sample["start_frame"] = record.start_frame
        sample["end_frame"] = record.end_frame
        sample["clip_start_frame"] = record.clip_start_frame
        sample["clip_end_frame"] = record.clip_end_frame
        sample["clip_start_second"] = record.clip_start_second
        sample["clip_end_second"] = record.clip_end_second
        sample["video_name"] = record.label
        return sample

    def _get(self, record):

        i3d_feature = h5py.File(record.path, 'r')
        vid = record.label
        feature = torch.tensor(i3d_feature[vid]['i3d_rgb_features'])

        return feature, min(len(feature), self.max_fms)

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
