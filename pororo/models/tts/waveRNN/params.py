class HParams:
    # CONFIG -----------------------------------------------------------------------------------------------------------#

    # Here are the input and output data paths (Note: you can override wav_path in preprocess.py)
    data_path = "dataset/"

    # model ids are separate - that way you can use a new tts with an old wavernn and vice versa
    # NB: expect undefined behaviour if models were trained on different DSP settings
    voc_model_id = "css_raw"

    # DSP --------------------------------------------------------------------------------------------------------------#

    # Settings for all models
    sample_rate = 22050
    n_fft = 2048
    fft_bins = n_fft // 2 + 1
    num_mels = 80
    hop_length = 275  # 12.5ms - in line with Tacotron 2 paper
    win_length = 1100  # 50ms - same reason as above
    fmin = 40
    min_level_db = -100
    ref_level_db = 20
    bits = 10  # bit depth of signal
    mu_law = (
        True  # Recommended to suppress noise if using raw bits in hp.voc_mode below
    )
    peak_norm = False  # Normalise to the peak of each wav file

    # WAVERNN / VOCODER ------------------------------------------------------------------------------------------------#

    # Model Hparams
    voc_mode = "RAW"  # either 'RAW' (softmax on raw bits) or 'MOL' (sample from mixture of logistics)
    voc_upsample_factors = (
        5,
        5,
        11,
    )  # NB - this needs to correctly factorise hop_length
    voc_rnn_dims = 512
    voc_fc_dims = 512
    voc_compute_dims = 128
    voc_res_out_dims = 128
    voc_res_blocks = 10

    # Training
    voc_batch_size = 64
    voc_lr = 1e-3
    lr_decay = 0.5
    lr_decay_start = 100000
    lr_decay_each = 100000
    weight_decay = 1e-6
    voc_checkpoint_every = 25_000
    voc_gen_at_checkpoint = 10  # number of samples to generate at each checkpoint
    voc_total_steps = 1_000_000  # Total number of training steps
    voc_test_samples = 50  # How many unseen samples to put aside for testing
    voc_pad = 2  # this will pad the input so that the resnet can 'see' wider than input length
    voc_seq_len = hop_length * 5  # must be a multiple of hop_length
    voc_clip_grad_norm = 4  # set to None if no gradient clipping needed

    # Generating / Synthesizing
    voc_gen_batched = True  # very fast (realtime+) single utterance batched generation
    voc_target = 11_000  # target number of samples to be generated in each batch entry
    voc_overlap = 550  # number of samples for crossfading between batches


hp = HParams()
