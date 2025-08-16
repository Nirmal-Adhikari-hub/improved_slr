from torch.utils.data import DataLoader
from cslr.data_loader.phoenix_feeder import PhoenixFeeder, make_collate_fn



def build_loaders(cfg, kernel_spec):
    d = cfg.data
    is_video = d.get("datatype", "video") == "video"

    train_set = PhoenixFeeder(
        dataset_root=d.dataset_root,
        preprocess_root=d.preprocess_root,
        gloss_dict_path=d.gloss_dict_path,
        dataset=d.get("dataset", "phoenix2014"),
        mode="train",
        datatype=d.get("datatype", "video"),
        frame_interval=d.get("frame_interval", 1),
        image_scale=d.get("image_scale", 1.0),
        input_size=d.get("input_size", 224),
        kernel_spec=kernel_spec,
        transform_train=True,
        frame_subdir=d.get("frame_subdir", "features/fullFrame-256x256px"),
    )

    dev_set = PhoenixFeeder(
        dataset_root=d.dataset_root,
        preprocess_root=d.preprocess_root,
        gloss_dict_path=d.gloss_dict_path,
        dataset=d.get("dataset", "phoenix2014"),
        mode="dev",
        datatype=d.get("datatype", "video"),
        frame_interval=d.get("frame_interval", 1),
        image_scale=d.get("image_scale", 1.0),
        input_size=d.get("input_size", 224),
        kernel_spec=kernel_spec,
        transform_train=False,
        frame_subdir=d.get("frame_subdir", "features/fullFrame-256x256px"),
    )

    collate = make_collate_fn(
        kernel_spec,
        is_video=is_video,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=d.batch_size,
        shuffle=True,
        num_workers=d.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )

    dev_loader = DataLoader(
        dev_set,
        batch_size=d.batch_size,
        shuffle=False,
        num_workers=d.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )

    return train_loader, dev_loader
