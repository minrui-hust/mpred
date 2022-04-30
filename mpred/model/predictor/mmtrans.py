import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from mai.utils import FI
from mai.model import BaseModule
from mai.model.utils.construct_mask import construct_mask
from mai.utils.misc import is_nan_or_inf


@FI.register
class MMTrans(BaseModule):
    def __init__(self,
                 hparam,
                 agent_enc,
                 agent_dec,
                 lane_net=None,
                 lane_enc=None,
                 lane_dec=None,
                 object_net=None,
                 object_enc=None,
                 object_dec=None,
                 head=None,
                 freeze_agent=False,
                 freeze_lane=False,
                 ):
        super().__init__()

        self.hparam = hparam
        self.lane_enable = hparam['lane_enable']
        self.object_enable = hparam['object_enable']
        self.output_stage = hparam['output_stage']
        self.additional_fusion = hparam.get('additional_fusion', False)
        self.time_shift_enable = hparam.get('time_shift_enable', False)
        self.freeze_agent = freeze_agent
        self.freeze_lane = freeze_lane
        model_dim = hparam['model_dim']
        lane_enc_dim = hparam['lane_enc_dim']
        object_enc_dim = hparam['object_enc_dim']
        agent_dim = hparam['agent_dim']
        K = hparam['num_queries']
        dropout = hparam['dropout']

        self.agent_emb = LinearEmbedding(agent_dim, model_dim)
        self.agent_pos_enc = PositionalEncoding(model_dim, dropout)
        self.agent_enc = FI.create(agent_enc)
        self.agent_dec = FI.create(agent_dec)
        self.agent_head = FI.create(head)

        if self.lane_enable:
            self.lane_emb = LinearEmbedding(lane_enc_dim, model_dim)
            self.lane_net = FI.create(lane_net)
            self.lane_enc = FI.create(lane_enc)
            self.lane_dec = FI.create(lane_dec)
            self.lane_head = FI.create(head)

        if self.object_enable:
            self.object_emb = LinearEmbedding(object_enc_dim, model_dim)
            self.object_net = FI.create(object_net)
            self.object_enc = FI.create(object_enc)
            self.object_dec = FI.create(object_dec)
            self.object_head = FI.create(head)

        if self.additional_fusion:
            self.additional_lane_dec = FI.create(lane_dec)

        self.query = nn.Embedding(K, model_dim)

        self.initialize()

    def initialize(self):
        for _, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

        #  self.query.weight.requires_grad = False
        nn.init.orthogonal_(self.query.weight)

        if self.freeze_agent:
            self.agent_emb.freeze()
            self.agent_pos_enc.freeze()
            self.agent_enc.freeze()
            self.agent_dec.freeze()

        if self.freeze_lane:
            self.lane_emb.freeze()
            self.lane_net.freeze()
            self.lane_enc.freeze()
            self.lane_dec.freeze()

    def forward_input(self, input):
        agent = input['agent']  # (B, 19, 4)
        object = input['object']  # (B, O, 19, 4)
        lane = input['lane']  # (B, L, 15, 7)
        object_num = input['object_num']  # (B)
        lane_num = input['lane_num']  # (B)

        # batch_size, max_agent_num, max_lane_num
        B, O, L, K = agent.shape[0], object.shape[1], lane.shape[1], self.hparam['num_queries']

        # (B, K, model_dim)
        query_batches = self.query.weight.unsqueeze(0).expand(B, -1, -1)

        out = {}

        # fusion agent
        agent = self.agent_emb(agent)
        agent = self.agent_pos_enc(agent)
        agent = self.agent_enc(agent)
        agent_out = self.agent_dec(query_batches, agent)
        if 'agent' in self.output_stage:
            agent_head_out = self.agent_head(agent_out)
            out['agent'] = agent_head_out

        # fusion lane
        if self.lane_enable:
            lane_mask = construct_mask(lane_num, L, inverse=True)
            lane = self.lane_net(lane)  # (B, L, 64)
            lane = self.lane_emb(lane)  # (B, L, model_dim)
            lane = self.lane_enc(lane, mask=lane_mask)  # (B, L, model_dim)
            lane_out = self.lane_dec(agent_out, lane, mask=lane_mask)
            if 'lane' in self.output_stage:
                lane_head_out = self.lane_head(lane_out)
                out['lane'] = lane_head_out

        if self.object_enable:
            object_mask = construct_mask(object_num, O, inverse=True)
            object = self.object_net(object)
            object = self.object_emb(object)
            object = self.object_enc(object, mask=object_mask)
            object_out = self.object_dec(lane_out, object, mask=object_mask)
            if self.additional_fusion:
                object_out = self.additional_lane_dec(
                    object_out, lane, mask=lane_mask)
            if 'object' in self.output_stage:
                object_head_out = self.object_head(object_out)
                out['object'] = object_head_out

        return out

    def forward_train(self, batch):
        out = {}
        out['output'] = self.forward_input(batch['input'])
        if self.time_shift_enable and 'time_shifted_input' in batch:
            out['time_shifted_output'] = self.forward_input(
                batch['time_shifted_input'])
        return out

    def forward_export(self, agent, lane, object, object_num, lane_num):
        # batch_size, max_agent_num, max_lane_num
        B, O, L, K = agent.shape[0], object.shape[1], lane.shape[1], self.hparam['num_queries']

        lane_mask = construct_mask(lane_num, L, inverse=True)  # (B, L)

        # fusion agent
        agent = self.agent_emb(agent)

        agent = self.agent_pos_enc(agent)  # (B, 19, model_dim)

        # (B, 19, model_dim)
        agent = self.agent_enc(agent)

        # (B, K, model_dim)
        query_batches = self.query.weight.unsqueeze(0).expand(B, -1, -1)

        # (B, K, model_dim)
        agent_out = self.agent_dec(query_batches, agent)

        # fusion lane
        if self.lane_enable:
            lane = self.lane_net(lane)  # (B, L, 64)

            lane = self.lane_emb(lane)  # (B, L, model_dim)

            lane = self.lane_enc(lane, mask=lane_mask)  # (B, L, model_dim)

            # (B, K, model_dim)
            lane_out = self.lane_dec(agent_out, lane, mask=lane_mask)

        if self.object_enable:
            raise NotImplementedError

        if not self.lane_enable and not self.object_enable:
            head_in = agent_out
        elif not self.object_enable:
            head_in = lane_out
        else:
            raise NotImplementedError

        head_out = self.head(head_in)

        return head_out


@FI.register
class LaneNet(BaseModule):
    def __init__(self, in_channels, hidden_unit, layer_num, act_cfg=dict(type='ReLU')):
        super(LaneNet, self).__init__()
        self.layer_list = nn.ModuleList()
        for i in range(layer_num):
            self.layer_list.append(FI.create(dict(
                type='MLP',
                in_channels=in_channels,
                hidden_channels=hidden_unit,
                out_channels=hidden_unit,
                act_cfg=act_cfg,
            )))
            in_channels = hidden_unit*2

    def forward(self, x):
        '''
            Extract lane_feature from vectorized lane representation

        Args:
            lane: [batch size, max_lane_num, 9, 7] (vectorized representation)

        Returns:
            x_max: [batch size, max_lane_num, 64]
        '''
        for layer in self.layer_list:
            # x [bs,max_lane_num,9,dim]
            x = layer(x)
            x_max = torch.max(x, -2)[0]
            x_max_expand = x_max.unsqueeze(2).repeat(1, 1, x.shape[2], 1)
            x = torch.cat([x, x_max_expand], dim=-1)
        return x_max


class LinearEmbedding(BaseModule):
    def __init__(self, inp_size, d_model):
        super(LinearEmbedding, self).__init__()
        # lut => lookup table
        self.lut = nn.Linear(inp_size, d_model, bias=True)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(BaseModule):
    """
    Implement the PE function.
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        x = x + self.pe[:x.shape[-2]]
        return self.dropout(x)
