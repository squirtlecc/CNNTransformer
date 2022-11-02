from torch.utils.data import DataLoader

from utils.register import Register, buildFromConfig
# from mmcv.parallel import collate

DATASETS = Register('datasets')
PIPELINES = Register('pipelines')


def build(cfgs, register, default_config=None):
    return buildFromConfig(cfgs, register, default_config)


def buildDataset(split_cfg, cfg):
    return build(split_cfg, DATASETS, default_config=dict(dataset_cfg=cfg))


def buildDataloader(register_cfg, dataset_cfg, is_train=True) -> DataLoader:
    shuffle = is_train

    datasets = buildDataset(register_cfg, dataset_cfg)

    # samples_per_gpu = dataset_cfg.batch_size // dataset_cfg.gpus

    # fixed img_metas issues
    def collate_fn(batchs):
        from torch.utils.data.dataloader import default_collate
        metas = [batch.pop('img_metas') for batch in batchs]

        batchs = default_collate(batchs)
        batchs['img_metas'] = metas

        return batchs

    data_loader = DataLoader(
        datasets,
        batch_size=dataset_cfg.batch_size,
        shuffle=shuffle,
        num_workers=dataset_cfg.workers,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False
    )
    return data_loader
