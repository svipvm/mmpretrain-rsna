
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(896, 1280), keep_ratio=False),
    dict(type='PackInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    dataset=dict(
        type="ImageNet",
        test_mode=True,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# Model settings
model = dict(
    type='ImageClassifier',
    data_preprocessor=dict(
        type='ClsDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
    ),
    # ResNet
    backbone=dict(
        type='ResNet',
        depth=50,
        out_indices=(3,),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', 
            checkpoint='torchvision://resnet50'
        )
    ),
    # Neck
    neck=dict(type='GlobalMaxPooling'),
    # Head
    head=dict(
        type='LinearClsHead', 
        num_classes=1,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)
