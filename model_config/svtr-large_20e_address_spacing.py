addres_train_list = [
    dict(
        ann_file='train_label.json',
        data_root='/data/ocr/synth30k_new/reco',
        pipeline=None,
        type='OCRDataset'),
    dict(
        ann_file='train_label.json',
        data_root='/data/ocr/synth10k_pp_random/reco',
        pipeline=None,
        type='OCRDataset'),
    dict(
        ann_file='train_label.json',
        data_root='/data/ocr/merged_synth_doc_cosmetic_medicine_36k/reco',
        pipeline=None,
        type='OCRDataset'),
    dict(
        ann_file='annotation_RecogDataSet.json',
        data_root='/data/ocr/synth_recog_dataset',
        pipeline=None,
        type='OCRDataset'),
    dict(
        ann_file='annotation_pp_spacing.json',
        data_root='/data/ocr/pp_spacing_word',
        pipeline=None,
        type='OCRDataset'),
    dict(
        ann_file='train_label.json',
        data_root='/data/ocr/ko_synthTIGER_100M_address_v1',
        pipeline=None,
        type='OCRDataset'),
]
cosmetic_textrecog_data_root = '/data/ocr/merged_synth_doc_cosmetic_medicine_36k/reco'
cosmetic_textrecog_test = dict(
    ann_file='val_label.json',
    data_root='/data/ocr/merged_synth_doc_cosmetic_medicine_36k/reco',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
cosmetic_textrecog_train = dict(
    ann_file='train_label.json',
    data_root='/data/ocr/merged_synth_doc_cosmetic_medicine_36k/reco',
    pipeline=None,
    type='OCRDataset')
cubox_textrecog_data_root = '/data/ocr/cubox_testset/CUBOX_KR_Recog'
cubox_textrecog_test = dict(
    ann_file='CUBOX_annotation_refined.json',
    data_root='/data/ocr/cubox_testset/CUBOX_KR_Recog',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
cute80_textrecog_data_root = '/data/ocr/cute80'
cute80_textrecog_test = dict(
    ann_file='textrecog_test.json',
    data_root='/data/ocr/cute80',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
default_hooks = dict(
    checkpoint=dict(interval=5, type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffer=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw_gt=False,
        draw_pred=False,
        enable=False,
        interval=1,
        show=False,
        type='VisualizationHook'))
default_scope = 'mmocr'
dictionary = dict(
    dict_file=
    'dicts/korean_english_digits_symbols.txt',
    type='Dictionary',
    with_padding=True,
    with_unknown=True)
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
icdar2013_857_textrecog_test = dict(
    ann_file='textrecog_test_857.json',
    data_root='/data/ocr/icdar_reco/icdar2013',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
icdar2013_textrecog_data_root = '/data/ocr/icdar_reco/icdar2013'
icdar2013_textrecog_test = dict(
    ann_file='textrecog_test.json',
    data_root='/data/ocr/icdar_reco/icdar2013',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
icdar2013_textrecog_train = dict(
    ann_file='textrecog_train.json',
    data_root='/data/ocr/icdar_reco/icdar2013',
    pipeline=None,
    type='OCRDataset')
icdar2015_1811_textrecog_test = dict(
    ann_file='textrecog_test_1811.json',
    data_root='/data/ocr/icdar_reco/icdar2015',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
icdar2015_textrecog_data_root = '/data/ocr/icdar_reco/icdar2015'
icdar2015_textrecog_test = dict(
    ann_file='textrecog_test.json',
    data_root='/data/ocr/icdar_reco/icdar2015',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
icdar2015_textrecog_train = dict(
    ann_file='textrecog_train.json',
    data_root='/data/ocr/icdar_reco/icdar2015',
    pipeline=None,
    type='OCRDataset')
iiit5k_textrecog_data_root = '/data/ocr/iiit5k'
iiit5k_textrecog_test = dict(
    ann_file='textrecog_test.json',
    data_root='/data/ocr/iiit5k',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
iiit5k_textrecog_train = dict(
    ann_file='textrecog_train.json',
    data_root='/data/ocr/iiit5k',
    pipeline=None,
    type='OCRDataset')
ko_synthTIGER_100M_address_v1_root = '/data/ocr/ko_synthTIGER_100M_address_v1'
ko_synthTIGER_100M_address_v1_train = dict(
    ann_file='train_label.json',
    data_root='/data/ocr/ko_synthTIGER_100M_address_v1',
    pipeline=None,
    type='OCRDataset')
launcher = 'pytorch'
load_from = 'pretrained/svtr-large_20e_mj1000k_pretrain/epoch_20.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=10)
mjsynth_data_root = '/data/ocr/mjsynth_synthTIGER_100M_v2'
mjsynth_tiger_original_root = '/data/ocr/synthTIGER_v1.1/synthtiger_v1.1'
mjsynth_tiger_original_train = dict(
    ann_file='train_label.json',
    data_root='/data/ocr/synthTIGER_v1.1/synthtiger_v1.1',
    pipeline=None,
    type='OCRDataset')
mjsynth_tiger_train = dict(
    ann_file='annotation.json',
    data_root='/data/ocr/mjsynth_synthTIGER_100M_v2',
    pipeline=None,
    type='OCRDataset')
model = dict(
    data_preprocessor=dict(
        mean=[
            127.5,
        ], std=[
            127.5,
        ], type='TextRecogDataPreprocessor'),
    decoder=dict(
        dictionary=dict(
            dict_file=
            'dicts/korean_english_digits_symbols.txt',
            type='Dictionary',
            with_padding=True,
            with_unknown=True),
        in_channels=384,
        module_loss=dict(
            letter_case='unchanged', type='CTCModuleLoss', zero_infinity=True),
        postprocessor=dict(type='CTCPostProcessor'),
        type='SVTRDecoder'),
    encoder=dict(
        depth=[
            3,
            9,
            9,
        ],
        embed_dims=[
            192,
            256,
            512,
        ],
        img_size=[
            48,
            160,
        ],
        in_channels=3,
        max_seq_len=40,
        merging_types='Conv',
        mixer_types=[
            'Local',
            'Local',
            'Local',
            'Local',
            'Local',
            'Local',
            'Local',
            'Local',
            'Local',
            'Local',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global',
        ],
        num_heads=[
            6,
            8,
            16,
        ],
        out_channels=384,
        prenorm=False,
        type='SVTREncoder',
        window_size=[
            [
                7,
                11,
            ],
            [
                7,
                11,
            ],
            [
                7,
                11,
            ],
        ]),
    preprocessor=dict(
        in_channels=3,
        margins=[
            0.05,
            0.05,
        ],
        num_control_points=20,
        output_image_size=(
            48,
            160,
        ),
        resized_image_size=(
            32,
            64,
        ),
        type='STN'),
    type='SVTR')
nia_food_menu_data_root = '/data/ocr/nia/222.food_menu/reco'
nia_food_menu_test = dict(
    ann_file='val_label.json',
    data_root='/data/ocr/nia/222.food_menu/reco',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
nia_food_menu_train = dict(
    ann_file='train_label.json',
    data_root='/data/ocr/nia/222.food_menu/reco',
    pipeline=None,
    type='OCRDataset')
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.99,
        ),
        eps=8e-08,
        lr=0.00025,
        type='AdamW',
        weight_decay=0.05),
    type='OptimWrapper')
param_scheduler = [
    dict(
        convert_to_iter_based=True,
        end=2,
        end_factor=1.0,
        start_factor=0.5,
        type='LinearLR',
        verbose=False),
    dict(
        T_max=19,
        begin=2,
        convert_to_iter_based=True,
        end=20,
        type='CosineAnnealingLR',
        verbose=False),
]
pp_spacing_word_root = '/data/ocr/pp_spacing_word'
pp_spacing_word_train = dict(
    ann_file='annotation_pp_spacing.json',
    data_root='/data/ocr/pp_spacing_word',
    pipeline=None,
    type='OCRDataset')
randomness = dict(seed=None)
resume = False
svt_textrecog_data_root = '/data/ocr/svt'
svt_textrecog_test = dict(
    ann_file='textrecog_test.json',
    data_root='/data/ocr/svt',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
svt_textrecog_train = dict(
    ann_file='textrecog_train.json',
    data_root='/data/ocr/svt',
    pipeline=None,
    type='OCRDataset')
svtp_textrecog_data_root = '/data/ocr/svtp'
svtp_textrecog_test = dict(
    ann_file='textrecog_test.json',
    data_root='/data/ocr/svtp',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
svtp_textrecog_train = dict(
    ann_file='textrecog_train.json',
    data_root='/data/ocr/svtp',
    pipeline=None,
    type='OCRDataset')
synth10k_textrecog_data_root = '/data/ocr/synth10k_pp_random/reco'
synth10k_textrecog_test = dict(
    ann_file='val_label.json',
    data_root='/data/ocr/synth10k_pp_random/reco',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
synth10k_textrecog_train = dict(
    ann_file='train_label.json',
    data_root='/data/ocr/synth10k_pp_random/reco',
    pipeline=None,
    type='OCRDataset')
synth30k_new_textrecog_data_root = '/data/ocr/synth30k_new/reco'
synth30k_new_textrecog_test = dict(
    ann_file='val_label.json',
    data_root='/data/ocr/synth30k_new/reco',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
synth30k_new_textrecog_train = dict(
    ann_file='train_label.json',
    data_root='/data/ocr/synth30k_new/reco',
    pipeline=None,
    type='OCRDataset')
synth_text_recog_data_root = '/data/ocr/synth_recog_dataset'
synth_text_recog_data_train = dict(
    ann_file='annotation_RecogDataSet.json',
    data_root='/data/ocr/synth_recog_dataset',
    pipeline=None,
    type='OCRDataset')
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=128,
    dataset=dict(
        datasets=[
            dict(
                ann_file='CUBOX_annotation_refined.json',
                data_root='/data/ocr/cubox_testset/CUBOX_KR_Recog',
                pipeline=None,
                test_mode=True,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=(
                256,
                64,
            ), type='Resize'),
            dict(type='LoadOCRAnnotations', with_text=True),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'valid_ratio',
                ),
                type='PackTextRecogInputs'),
        ],
        type='ConcatDataset'),
    drop_last=False,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    dataset_prefixes=[
        '30',
    ],
    metrics=[
        dict(type='OneMinusNEDMetric'),
        dict(
            mode=[
                'exact',
                'ignore_case',
                'ignore_case_symbol',
            ],
            type='WordMetric'),
        dict(type='CharMetric'),
    ],
    type='MultiDatasetsEvaluator')
test_list = [
    dict(
        ann_file='CUBOX_annotation_refined.json',
        data_root='/data/ocr/cubox_testset/CUBOX_KR_Recog',
        pipeline=None,
        test_mode=True,
        type='OCRDataset'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(scale=(
        256,
        64,
    ), type='Resize'),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
            'valid_ratio',
        ),
        type='PackTextRecogInputs'),
]
train_cfg = dict(max_epochs=20, type='EpochBasedTrainLoop', val_interval=5)
train_dataloader = dict(
    batch_size=512,
    dataset=dict(
        datasets=[
            dict(
                ann_file='train_label.json',
                data_root='/data/ocr/synth30k_new/reco',
                pipeline=None,
                type='OCRDataset'),
            dict(
                ann_file='train_label.json',
                data_root='/data/ocr/synth10k_pp_random/reco',
                pipeline=None,
                type='OCRDataset'),
            dict(
                ann_file='train_label.json',
                data_root=
                '/data/ocr/merged_synth_doc_cosmetic_medicine_36k/reco',
                pipeline=None,
                type='OCRDataset'),
            dict(
                ann_file='annotation_RecogDataSet.json',
                data_root='/data/ocr/synth_recog_dataset',
                pipeline=None,
                type='OCRDataset'),
            dict(
                ann_file='annotation_pp_spacing.json',
                data_root='/data/ocr/pp_spacing_word',
                pipeline=None,
                type='OCRDataset'),
            dict(
                ann_file='train_label.json',
                data_root='/data/ocr/ko_synthTIGER_100M_address_v1',
                pipeline=None,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(ignore_empty=True, min_size=5, type='LoadImageFromFile'),
            dict(type='LoadOCRAnnotations', with_text=True),
            dict(
                prob=0.4,
                transforms=[
                    dict(type='TextRecogGeneralAug'),
                ],
                type='RandomApply'),
            dict(
                prob=0.4,
                transforms=[
                    dict(type='CropHeight'),
                ],
                type='RandomApply'),
            dict(
                condition='min(results["img_shape"])>10',
                true_transforms=dict(
                    prob=0.4,
                    transforms=[
                        dict(
                            kernel_size=5,
                            op='GaussianBlur',
                            sigma=1,
                            type='TorchVisionWrapper'),
                    ],
                    type='RandomApply'),
                type='ConditionApply'),
            dict(
                prob=0.4,
                transforms=[
                    dict(
                        brightness=0.5,
                        contrast=0.5,
                        hue=0.1,
                        op='ColorJitter',
                        saturation=0.5,
                        type='TorchVisionWrapper'),
                ],
                type='RandomApply'),
            dict(
                prob=0.4,
                transforms=[
                    dict(type='ImageContentJitter'),
                ],
                type='RandomApply'),
            dict(
                prob=0.4,
                transforms=[
                    dict(
                        args=[
                            dict(
                                cls='AdditiveGaussianNoise',
                                scale=0.31622776601683794),
                        ],
                        type='ImgAugWrapper'),
                ],
                type='RandomApply'),
            dict(
                prob=0.4,
                transforms=[
                    dict(type='ReversePixels'),
                ],
                type='RandomApply'),
            dict(scale=(
                256,
                64,
            ), type='Resize'),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'valid_ratio',
                ),
                type='PackTextRecogInputs'),
        ],
        type='ConcatDataset'),
    num_workers=24,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_list = [
    dict(
        ann_file='train_label.json',
        data_root='/data/ocr/nia/222.food_menu/reco',
        pipeline=None,
        type='OCRDataset'),
    dict(
        ann_file='val_label.json',
        data_root='/data/ocr/nia/222.food_menu/reco',
        pipeline=None,
        test_mode=True,
        type='OCRDataset'),
    dict(
        ann_file='train_label.json',
        data_root='/data/ocr/synth30k_new/reco',
        pipeline=None,
        type='OCRDataset'),
    dict(
        ann_file='val_label.json',
        data_root='/data/ocr/synth30k_new/reco',
        pipeline=None,
        test_mode=True,
        type='OCRDataset'),
    dict(
        ann_file='train_label.json',
        data_root='/data/ocr/synth10k_pp_random/reco',
        pipeline=None,
        type='OCRDataset'),
    dict(
        ann_file='val_label.json',
        data_root='/data/ocr/synth10k_pp_random/reco',
        pipeline=None,
        test_mode=True,
        type='OCRDataset'),
    dict(
        ann_file='train_label.json',
        data_root='/data/ocr/merged_synth_doc_cosmetic_medicine_36k/reco',
        pipeline=None,
        type='OCRDataset'),
    dict(
        ann_file='annotation_RecogDataSet.json',
        data_root='/data/ocr/synth_recog_dataset',
        pipeline=None,
        type='OCRDataset'),
    dict(
        ann_file='textrecog_test.json',
        data_root='/data/ocr/cute80',
        pipeline=None,
        test_mode=True,
        type='OCRDataset'),
    dict(
        ann_file='textrecog_train.json',
        data_root='/data/ocr/icdar_reco/icdar2015',
        pipeline=None,
        type='OCRDataset'),
    dict(
        ann_file='textrecog_test.json',
        data_root='/data/ocr/icdar_reco/icdar2015',
        pipeline=None,
        test_mode=True,
        type='OCRDataset'),
    dict(
        ann_file='textrecog_train.json',
        data_root='/data/ocr/icdar_reco/icdar2013',
        pipeline=None,
        type='OCRDataset'),
    dict(
        ann_file='textrecog_test.json',
        data_root='/data/ocr/icdar_reco/icdar2013',
        pipeline=None,
        test_mode=True,
        type='OCRDataset'),
    dict(
        ann_file='textrecog_train.json',
        data_root='/data/ocr/iiit5k',
        pipeline=None,
        type='OCRDataset'),
    dict(
        ann_file='textrecog_test.json',
        data_root='/data/ocr/iiit5k',
        pipeline=None,
        test_mode=True,
        type='OCRDataset'),
    dict(
        ann_file='textrecog_train.json',
        data_root='/data/ocr/svt',
        pipeline=None,
        type='OCRDataset'),
    dict(
        ann_file='textrecog_test.json',
        data_root='/data/ocr/svt',
        pipeline=None,
        test_mode=True,
        type='OCRDataset'),
    dict(
        ann_file='textrecog_test.json',
        data_root='/data/ocr/svtp',
        pipeline=None,
        test_mode=True,
        type='OCRDataset'),
    dict(
        ann_file='annotation_pp_spacing.json',
        data_root='/data/ocr/pp_spacing_word',
        pipeline=None,
        type='OCRDataset'),
    dict(
        ann_file='train_label.json',
        data_root='/data/ocr/ko_synthTIGER_100M_address_v1',
        pipeline=None,
        type='OCRDataset'),
]
train_pipeline = [
    dict(ignore_empty=True, min_size=5, type='LoadImageFromFile'),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        prob=0.4,
        transforms=[
            dict(type='TextRecogGeneralAug'),
        ],
        type='RandomApply'),
    dict(prob=0.4, transforms=[
        dict(type='CropHeight'),
    ], type='RandomApply'),
    dict(
        condition='min(results["img_shape"])>10',
        true_transforms=dict(
            prob=0.4,
            transforms=[
                dict(
                    kernel_size=5,
                    op='GaussianBlur',
                    sigma=1,
                    type='TorchVisionWrapper'),
            ],
            type='RandomApply'),
        type='ConditionApply'),
    dict(
        prob=0.4,
        transforms=[
            dict(
                brightness=0.5,
                contrast=0.5,
                hue=0.1,
                op='ColorJitter',
                saturation=0.5,
                type='TorchVisionWrapper'),
        ],
        type='RandomApply'),
    dict(
        prob=0.4,
        transforms=[
            dict(type='ImageContentJitter'),
        ],
        type='RandomApply'),
    dict(
        prob=0.4,
        transforms=[
            dict(
                args=[
                    dict(
                        cls='AdditiveGaussianNoise',
                        scale=0.31622776601683794),
                ],
                type='ImgAugWrapper'),
        ],
        type='RandomApply'),
    dict(
        prob=0.4,
        transforms=[
            dict(type='ReversePixels'),
        ],
        type='RandomApply'),
    dict(scale=(
        256,
        64,
    ), type='Resize'),
    dict(
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
            'valid_ratio',
        ),
        type='PackTextRecogInputs'),
]
tta_model = dict(type='EncoderDecoderRecognizerTTAModel')
tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(
                    condition="results['img_shape'][1]<results['img_shape'][0]",
                    true_transforms=[
                        dict(
                            args=[
                                dict(cls='Rot90', k=0, keep_size=False),
                            ],
                            type='ImgAugWrapper'),
                    ],
                    type='ConditionApply'),
                dict(
                    condition="results['img_shape'][1]<results['img_shape'][0]",
                    true_transforms=[
                        dict(
                            args=[
                                dict(cls='Rot90', k=1, keep_size=False),
                            ],
                            type='ImgAugWrapper'),
                    ],
                    type='ConditionApply'),
                dict(
                    condition="results['img_shape'][1]<results['img_shape'][0]",
                    true_transforms=[
                        dict(
                            args=[
                                dict(cls='Rot90', k=3, keep_size=False),
                            ],
                            type='ImgAugWrapper'),
                    ],
                    type='ConditionApply'),
            ],
            [
                dict(scale=(
                    256,
                    64,
                ), type='Resize'),
            ],
            [
                dict(type='LoadOCRAnnotations', with_text=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'valid_ratio',
                    ),
                    type='PackTextRecogInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=128,
    dataset=dict(
        datasets=[
            dict(
                ann_file='CUBOX_annotation_refined.json',
                data_root='/data/ocr/cubox_testset/CUBOX_KR_Recog',
                pipeline=None,
                test_mode=True,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=(
                256,
                64,
            ), type='Resize'),
            dict(type='LoadOCRAnnotations', with_text=True),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'valid_ratio',
                ),
                type='PackTextRecogInputs'),
        ],
        type='ConcatDataset'),
    drop_last=False,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    dataset_prefixes=[
        '30',
    ],
    metrics=[
        dict(type='OneMinusNEDMetric'),
        dict(
            mode=[
                'exact',
                'ignore_case',
                'ignore_case_symbol',
            ],
            type='WordMetric'),
        dict(type='CharMetric'),
    ],
    type='MultiDatasetsEvaluator')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    font_properties='fonts/NanumBarunGothic.ttf',
    name='visualizer',
    type='TextRecogLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/svtr-large_20e_address_spacing'
