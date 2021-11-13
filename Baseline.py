import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class Baseline(nn.Module):
    def __init__(self, config):
        super(Baseline, self).__init__()
        self.config = config
        self.slstm = nn.LSTM(input_size=config.sdim, hidden_size=config.slstm_hidden_dim)
        self.vfc = nn.Linear(in_features=config.vdim, out_features=config.v_out_dim)
        self.mixlstm = nn.LSTM(input_size=config.slstm_hidden_dim + config.v_out_dim, hidden_size=config.mixlstm_hidden_dim,
                               num_layers=2, bidirectional=True)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.mixtfm = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.score_predictor = nn.Sequential(
            nn.Linear(in_features=2 * config.mixlstm_hidden_dim, out_features=config.score_hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=config.score_hidden_dim, out_features=1),
            nn.Sigmoid(),
        )
        self.index_predictor = nn.Sequential(
            nn.Linear(in_features=config.max_fms, out_features=128),
            nn.Tanh(),
            nn.Linear(in_features=128, out_features=2),
        )

    def forward(self, visual_feature, text_feature, fms_list, len_list):
        vfs = F.relu(self.vfc(visual_feature))
        packed_sfs, (hn, cn) = self.slstm(pack_padded_sequence(text_feature.transpose(0, 1), len_list, enforce_sorted=False))
        # hn:1,b,d
        mix_in = torch.cat([vfs, hn.transpose(0, 1).repeat([1, max(fms_list), 1])], dim=-1)
        out, _ = self.mixlstm(pack_padded_sequence(mix_in.transpose(0, 1), fms_list, enforce_sorted=False))
        out, _ = pad_packed_sequence(out)
        # out = self.mixtfm(mix_in.transpose(0, 1))
        out = out.transpose(0, 1)
        sc = self.score_predictor(out).squeeze(-1)
        indexes = self.index_predictor(sc)
        return sc, indexes