from copy import deepcopy as _deepcopy
from mai.utils import GCFG

# global config maybe override by command line
batch_size = GCFG['batch_size'] or 32  # different from original, which is 4
num_workers = GCFG['num_workers'] or 2
max_epochs = GCFG['max_epochs'] or 128
lr_scale = GCFG['lr_scale'] or 1.0  # may rescale by gpu number
dataset_root = GCFG['dataset_root'] or '/data/waymo'

# global config
############################
output_stage = 'object'
lane_enable = True
object_enable = True
model_dim = 128
pos_dim = 64
dist_dim = 128
lane_enc_dim = 64
object_enc_dim = 64
agent_dim = 4
num_queries = 6
dropout = 0.0
num_heads = 2
agent_layers = 2
lane_layers = 2
object_layers = 2
pred_win = 30
obj_radius = 50


# model config
############################
model_train = dict(
    type='MMTrans',
    freeze_agent=False,
    freeze_lane=False,
    hparam=dict(
        output_stage=output_stage,
        lane_enable=lane_enable,
        object_enable=object_enable,
        model_dim=model_dim,
        pos_dim=pos_dim,
        dist_dim=dist_dim,
        lane_enc_dim=lane_enc_dim,
        object_enc_dim=object_enc_dim,
        agent_dim=agent_dim,
        dropout=dropout,
        num_queries=num_queries,
    ),
    agent_enc=dict(
        type='TransformerEncoder',
        layer_num=agent_layers,
        layer_cfg=dict(
            type='TransformerEncoderLayer',
            atten_cfg=dict(
                type='MultiHeadSelfAtten',
                embed_dim=model_dim,
                dropout=dropout,
                num_heads=num_heads,
            ),
            ff_cfg=dict(
                type='MLP',
                in_channels=model_dim,
                hidden_channels=model_dim*2,
                out_channels=model_dim,
                norm_cfg=None,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            norm_cfg=dict(
                type='LayerNorm',
                normalized_shape=model_dim,
            ),
        ),
    ),
    agent_dec=dict(
        type='TransformerDecoder',
        layer_num=agent_layers,
        layer_cfg=dict(
            type='TransformerDecoderLayer',
            self_atten_cfg=dict(
                type='MultiHeadSelfAtten',
                embed_dim=model_dim,
                dropout=dropout,
                num_heads=num_heads,
            ),
            cross_atten_cfg=dict(
                type='MultiHeadCrossAtten',
                embed_dim=model_dim,
                dropout=dropout,
                num_heads=num_heads,
            ),
            ff_cfg=dict(
                type='MLP',
                in_channels=model_dim,
                hidden_channels=model_dim*2,
                out_channels=model_dim,
                norm_cfg=None,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            norm_cfg=dict(
                type='LayerNorm',
                normalized_shape=model_dim,
            ),
        ),
    ),
    lane_net=dict(
        type='LaneNet',
        in_channels=7,
        hidden_unit=64,
        layer_num=2,
        act_cfg=dict(type='ReLU', inplace=True),
    ),
    lane_enc=dict(
        type='TransformerEncoder',
        layer_num=lane_layers,
        layer_cfg=dict(
            type='TransformerEncoderLayer',
            atten_cfg=dict(
                type='MultiHeadSelfAtten',
                embed_dim=model_dim,
                dropout=dropout,
                num_heads=num_heads,
            ),
            ff_cfg=dict(
                type='MLP',
                in_channels=model_dim,
                hidden_channels=model_dim*2,
                out_channels=model_dim,
                norm_cfg=None,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            norm_cfg=dict(
                type='LayerNorm',
                normalized_shape=model_dim,
            ),
        ),
    ),
    lane_dec=dict(
        type='TransformerDecoder',
        layer_num=lane_layers,
        layer_cfg=dict(
            type='TransformerDecoderLayer',
            self_atten_cfg=dict(
                type='MultiHeadSelfAtten',
                embed_dim=model_dim,
                dropout=dropout,
                num_heads=num_heads,
            ),
            cross_atten_cfg=dict(
                type='MultiHeadCrossAtten',
                embed_dim=model_dim,
                dropout=dropout,
                num_heads=num_heads,
            ),
            ff_cfg=dict(
                type='MLP',
                in_channels=model_dim,
                hidden_channels=model_dim*2,
                out_channels=model_dim,
                norm_cfg=None,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            norm_cfg=dict(
                type='LayerNorm',
                normalized_shape=model_dim,
            ),
        ),
    ),
    object_net=dict(
        type='LaneNet',
        in_channels=6,
        hidden_unit=64,
        layer_num=2,
        act_cfg=dict(type='ReLU', inplace=True),
    ),
    object_enc=dict(
        type='TransformerEncoder',
        layer_num=object_layers,
        layer_cfg=dict(
            type='TransformerEncoderLayer',
            atten_cfg=dict(
                type='MultiHeadSelfAtten',
                embed_dim=model_dim,
                dropout=dropout,
                num_heads=num_heads,
            ),
            ff_cfg=dict(
                type='MLP',
                in_channels=model_dim,
                hidden_channels=model_dim*2,
                out_channels=model_dim,
                norm_cfg=None,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            norm_cfg=dict(
                type='LayerNorm',
                normalized_shape=model_dim,
            ),
        ),
    ),
    object_dec=dict(
        type='TransformerDecoder',
        layer_num=object_layers,
        layer_cfg=dict(
            type='TransformerDecoderLayer',
            self_atten_cfg=dict(
                type='MultiHeadSelfAtten',
                embed_dim=model_dim,
                dropout=dropout,
                num_heads=num_heads,
            ),
            cross_atten_cfg=dict(
                type='MultiHeadCrossAtten',
                embed_dim=model_dim,
                dropout=dropout,
                num_heads=num_heads,
            ),
            ff_cfg=dict(
                type='MLP',
                in_channels=model_dim,
                hidden_channels=model_dim*2,
                out_channels=model_dim,
                norm_cfg=None,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            norm_cfg=dict(
                type='LayerNorm',
                normalized_shape=model_dim,
            ),
        ),
    ),
    head=dict(
        type='MLPHead',
        in_channels=model_dim,
        heads={
            'traj': (model_dim, pred_win*2),
            'score': (model_dim, 1),
        },
        act_cfg=dict(type='ReLU', inplace=True),
    ),
)

model_eval = _deepcopy(model_train)

model_export = _deepcopy(model_eval)


# codecs config
############################
codec_train = dict(
    type='MMTransCodec',
    encode_cfg=dict(
        encode_data=True,
        encode_anno=True,
    ),
    decode_cfg=dict(),
    loss_cfg=dict(
        delta=1.8,
        wgt={
            'traj': 1.0,
            'score': 1.0,
        }
    ),
)

codec_eval = _deepcopy(codec_train)

codec_export = _deepcopy(codec_eval)
codec_export['encode_cfg']['encode_anno'] = False


dataloader_train = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    pin_memory=False,
    dataset=dict(
        type='ArgoPredDataset',
        info_path=f'{dataset_root}/train_info.pkl',
        load_opt=dict(
            map_path=f'{dataset_root}/map_info.pkl',
            obs_len=20,
            pred_len=30,
        ),
        filters=[],
        transforms=[
            dict(type='ObjectRangeFilter', obj_radius=obj_radius),
            dict(type='Normlize'),
            dict(type='MpredMirrorFlip', mirror_prob=0.0, flip_prob=0.5),
        ],
    ),
)

dataloader_eval = _deepcopy(dataloader_train)
dataloader_eval['shuffle'] = False
dataloader_eval['dataset']['info_path'] = f'{dataset_root}/val_info.pkl'
dataloader_eval['dataset']['filters'] = []
dataloader_eval['dataset']['transforms'] = [
    dict(type='ObjectRangeFilter', obj_radius=obj_radius),
    dict(type='Normlize'),
]

dataloader_export = _deepcopy(dataloader_eval)


# fit config
############################
fit = dict(
    max_epochs=max_epochs,
    optimizer=dict(
        type='AdamW',
        weight_decay=0.01,
        betas=(0.9, 0.99),
    ),
    scheduler=dict(
        type='OneCycleLR',
        max_lr=0.0002,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=10.0,
        final_div_factor=0.4,
        pct_start=0.1,
    ),
    grad_clip=dict(type='norm', value=0.1),
)


runtime = dict(
    train=dict(
        logger=[
            dict(type='TensorBoardLogger', flush_secs=15),
            dict(type='CSVLogger', flush_logs_every_n_steps=50),
        ],
    ),
    eval=dict(evaluate_min_epoch=max_epochs-1),
    test=dict(),
)


# collect config
#################################
model = dict(
    train=model_train,
    eval=model_eval,
    export=model_export,
)

codec = dict(
    train=codec_train,
    eval=codec_eval,
    export=codec_export,
)

data = dict(
    train=dataloader_train,
    eval=dataloader_eval,
    export=dataloader_export,
)
