# replace imports:
from cslr.data_loader.phoenix_feeder import PhoenixFeeder, make_collate_fn
# (remove the old PhoenixDataset/pad_sequence_batch import)

# replace dataset + dataloaders:
train_set = PhoenixFeeder(
    dataset_root=cfg['data']['dataset_root'],
    preprocess_root=cfg['data']['preprocess_root'],
    gloss_dict_path=cfg['data']['gloss_dict_path'],
    dataset=cfg['data'].get('dataset','phoenix2014'),
    mode='train',
    datatype=cfg['data'].get('datatype','video'),
    frame_interval=cfg['data'].get('frame_interval',1),
    image_scale=cfg['data'].get('image_scale',1.0),
    input_size=cfg['data'].get('input_size',224),
    kernel_spec=cfg['data'].get('kernel_spec', ['K5','P2','K5','P2']),
    transform_train=True,
    frame_subdir=cfg['data'].get('frame_subdir', 'features/fullFrame-256x256px'),
)
dev_set = PhoenixFeeder(
    dataset_root=cfg['data']['dataset_root'],
    preprocess_root=cfg['data']['preprocess_root'],
    gloss_dict_path=cfg['data']['gloss_dict_path'],
    dataset=cfg['data'].get('dataset','phoenix2014'),
    mode='dev',
    datatype=cfg['data'].get('datatype','video'),
    frame_interval=cfg['data'].get('frame_interval',1),
    image_scale=cfg['data'].get('image_scale',1.0),
    input_size=cfg['data'].get('input_size',224),
    kernel_spec=cfg['data'].get('kernel_spec', ['K5','P2','K5','P2']),
    transform_train=False,
    frame_subdir=cfg['data'].get('frame_subdir', 'features/fullFrame-256x256px'),
)

collate = make_collate_fn(cfg['data'].get('kernel_spec', ['K5','P2','K5','P2']),
                          is_video=(cfg['data'].get('datatype','video') == 'video'))

train_loader = DataLoader(
    train_set,
    batch_size=cfg["data"]["batch_size"],
    shuffle=True,
    num_workers=cfg["data"]["num_workers"],
    pin_memory=True,
    collate_fn=collate,
)
dev_loader = DataLoader(
    dev_set,
    batch_size=cfg["data"]["batch_size"],
    shuffle=False,
    num_workers=cfg["data"]["num_workers"],
    pin_memory=True,
    collate_fn=collate,
)
