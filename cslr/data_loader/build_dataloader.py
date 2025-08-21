from torch.utils.data import DataLoader
from cslr.data_loader.phoenix_feeder import PhoenixFeeder, make_collate_fn



def build_loaders(cfg, kernel_spec):
    d = cfg.data
    is_video = d.get("datatype", "video") == "video"

    set_common = dict(
    dataset_root=d.dataset_root,
    preprocess_root=d.preprocess_root,
    gloss_dict_path=d.gloss_dict_path,
    dataset=d.get("dataset", "phoenix2014"),
    datatype=d.get("datatype", "video"),
    frame_interval=d.get("frame_interval", 1),
    image_scale=d.get("image_scale", 1.0),
    input_size=d.get("input_size", 224),
    kernel_spec=kernel_spec,
    frame_subdir=d.get("frame_subdir", "features/fullFrame-256x256px"),
        )
    
    train_set = PhoenixFeeder(mode="train", transform_train=True, **set_common)
    dev_set = PhoenixFeeder(mode="dev", transform_train=False, **set_common)
    test_set = PhoenixFeeder(mode="test", transform_train=False, **set_common)

    collate = make_collate_fn(kernel_spec, is_video=is_video)

    loader_common = dict(
        batch_size=d.get("batch_size", 2),
        num_workers=d.get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate,
    )

    train_loader = DataLoader(train_set, shuffle=True, **loader_common)
    dev_loader = DataLoader(dev_set, shuffle=False, **loader_common)
    test_loader = DataLoader(test_set, shuffle=False, **loader_common)

    return train_loader, dev_loader, test_loader