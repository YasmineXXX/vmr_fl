import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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

class CrossAttention(nn.Module):
    def __init__(self,in1_dim,in2_dim,hidden_dim,out_dim,dropout_rate=0.0,weights_only=False):
        super().__init__()
        self.attention_hidden_dim=hidden_dim
        self.in2_dim = in2_dim
        self.in1_dim = in1_dim
        self.weights_only=weights_only
        self.Q = nn.Linear(in_features=in1_dim, out_features=self.attention_hidden_dim,bias=False)
        self.K = nn.Linear(in_features=in2_dim, out_features=self.attention_hidden_dim,bias=False)
        self.V =nn.Linear(in_features=in1_dim,out_features=out_dim)
        self.dropout=nn.Dropout(dropout_rate)
    def forward(self,main_feature,guide_feature):
        q=self.Q(main_feature)
        k=self.K(guide_feature)
        v=self.V(self.dropout(main_feature))
        attention_value=torch.exp(torch.tanh((q*k).sum(dim=-1,keepdim=True)))
        if self.weights_only:
            return attention_value
        return attention_value*v

class Permute(nn.Module):
 def __init__(self, *args):
  super().__init__()
  self.order = args

 def forward(self, x:torch.Tensor):
  # 如果数据集最后一个batch样本数量小于定义的batch_batch大小，会出现mismatch问题。可以自己修改下，如只传入后面的shape，然后通过x.szie(0)，来输入。
  return x.permute(self.order)

class VideoConv(nn.Module):
    def __init__(self,config,**kwargs):
        self.v_dim = config.vdim
        self.s_dim= 300
        self.time_steps=config.max_fms
        self.v_att_out_dim=256
        self.att_hidden_dim=128
        self.s_lstm_hidden_dim=256
        self.mix_lstm_hidden_dim=256
        self.predictor_pre_hidden_dim = 128
        self.predictor_hidden_dim = 64
        self.droprate=0.2
        super().__init__()
        self.__dict__.update(kwargs)
        self.drop=nn.Dropout(self.droprate)
        self.guid=CrossAttention(in1_dim=self.v_dim,in2_dim=self.s_lstm_hidden_dim,hidden_dim=self.att_hidden_dim,
                                 out_dim=self.v_att_out_dim,weights_only=False)
        self.s_lstm=nn.LSTM(input_size=self.s_dim,hidden_size=self.s_lstm_hidden_dim)
        self.s_att=nn.Linear(in_features=self.s_lstm_hidden_dim,out_features=1)
        self.mix_lstm=nn.LSTM(input_size=(self.v_att_out_dim+self.s_lstm_hidden_dim),hidden_size=self.mix_lstm_hidden_dim,bidirectional=True)
        self.predictor_pre = nn.Sequential(
            Permute(0, 2, 1),
            nn.BatchNorm1d(2 * self.mix_lstm_hidden_dim),
            nn.ReLU(),
            Permute(0, 2, 1),
            nn.Linear(2 * self.mix_lstm_hidden_dim, self.predictor_pre_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.predictor_pre_hidden_dim, 1),
            nn.Sigmoid()
        )
        self.predictor = nn.Sequential(
            nn.Linear(self.time_steps, self.predictor_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.predictor_hidden_dim, 2)
        )

    def forward(self,vfs:torch.Tensor,sfs:torch.Tensor,vlens,slens):
        vfs=self.drop(vfs)
        sfs=self.drop(sfs)
        s_short_packed,_=self.s_lstm(pack_padded_sequence(sfs.transpose(0,1),slens,enforce_sorted=False))
        s_short,_=pad_packed_sequence(s_short_packed)
        #T,B,D
        s_att_value=self.s_att(s_short).transpose(0, 1)
        #B,T,1
        for i,l in enumerate(slens):
            s_att_value[i][l:][:]=-1e10
        s_att_value=torch.softmax(s_att_value,dim=1)
        s_attned=(s_att_value*s_short.transpose(0,1)).sum(dim=-2,keepdim=True)
        #B,1,D
        v_attned=self.guid(vfs,s_attned)
        #B,T,D
        mix_in=torch.cat([v_attned.transpose(0,1),s_attned.transpose(0,1).repeat([max(vlens),1,1])],dim=-1)
        mix_in=self.drop(mix_in)
        mix_out_packed,_=self.mix_lstm(pack_padded_sequence(mix_in,vlens,enforce_sorted=False))
        mix_out,_=pad_packed_sequence(mix_out_packed)
        mix_out=mix_out.transpose(0,1)
        #mix_out shape (B,T, num_directions * hidden_size):
        y=self.predictor_pre(mix_out).squeeze(-1)
        start_end_index=self.predictor(y)
        return y,start_end_index