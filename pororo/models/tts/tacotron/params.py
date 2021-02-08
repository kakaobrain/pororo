import json


class Params:
    version = "1.0"  # is used during training as name of checkpoints and Tensorboard logs (together with timestamp and reached loss)
    """
    **************** PARAMETERS OF TRAINING LOOP ****************
    """

    epochs = 300  # training epochs
    batch_size = 52  # batch size during training (is parallelization is True, each GPU has batch_size // num_gpus examples)
    # if using encoder_type 'convolutional' or 'generated', should be divisible by the number of languages
    learning_rate = 1e-3  # starting learning rate
    learning_rate_decay = 0.5  # decay multiplier used by step learning rate scheduler (use 0.5 for halving)
    learning_rate_decay_start = 15000  # number of training steps until the first lr decay, expected to be greater than learning_rate_decay_each
    learning_rate_decay_each = 15000  # size of the learning rate scheduler step in training steps, it decays lr every this number steps
    learning_rate_encoder = 1e-3  # initial learning rate of the encoder, just used if encoder_optimizer is set to True
    weight_decay = 1e-6  # L2 regularization
    encoder_optimizer = False  # if True, different learning rates are used for the encoder and decoder, the ecoder uses learning_rate_encoder at start
    max_output_length = 5000  # maximal number of frames produced by decoder, the number of frames is usualy much lower during synthesis
    gradient_clipping = 0.25  # gradient norm clipping
    reversal_gradient_clipping = 0.25  # used if reversal_classifier is True, clips gradients flowing from adversarial classifier to encoder
    guided_attention_loss = True  # if True, guided attention loss term is used
    guided_attention_steps = 20000  # number of training steps for which the guided attention loss term is used
    guided_attention_toleration = (
        0.25  # starting variance of the guided attention (i.e. diagonal toleration)
    )
    guided_attention_gain = (
        1.00025  # multiplier applied after every batch to guided_attention_toleration
    )
    constant_teacher_forcing = True  # if True, ground-truth frames are with probability teacher_forcing passed into decoder, cosine decay is used otherwise
    teacher_forcing = (
        1.0  # ratio of ground-truth frames, used if constant_teacher_forcing is True
    )
    teacher_forcing_steps = 100000  # used if constant_teacher_forcing is False, cosine decay spans this number of trainig steps starting at teacher_forcing_start_steps
    teacher_forcing_start_steps = (
        50000  # number of training steps after which the teacher forcing decay starts
    )
    checkpoint_each_epochs = 10  # save a checkpoint every this number epochs
    parallelization = True  # if True, DataParallel (parallel batch) is used, supports any number of GPUs
    """
    ******************* DATASET SPECIFICATION *******************
    """

    dataset = "ljspeech"  # one of: css10, ljspeech, vctk, my_blizzard, my_common_voice, mailabs, must have implementation in loaders.py
    cache_spectrograms = True  # if True, during iterating the dataset, it first tries to load spectrograms (mel or linear) from cached files
    languages = [
        "en-us"
    ]  # list of lnguages which will be loaded from the dataset, codes should correspond to
    # espeak format (see 'phonemize --help) in order support the converion to phonemes
    balanced_sampling = False  # enables balanced sampling per languages (not speakers), multi_language must be True
    perfect_sampling = False  # used just if balanced_sampling is True, should be used together with encoder_type 'convolutional' or 'generated'
    # if True, each language has the same number of samples and these samples are grouped, batch_size must be divisible
    # if False, samples are taken from the multinomial distr. with replacement
    """
    *************************** TEXT ****************************
    """

    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "  # supported input alphabet, used for computation of character embeddings
    # for lower-case russian, greek, latin and pinyin use " abcdefghijklmnopqrstuvwxyzçèéßäöōǎǐíǒàáǔüèéìūòóùúāēěīâêôûñőűабвгдежзийклмнопрстуфхцчшщъыьэюяё"
    case_sensitive = True  # if False, all characters are lowered before usage
    remove_multiple_wspaces = True  # if True, multiple whitespaces, leading and trailing whitespaces, etc. are removed
    use_punctuation = (
        True  # if True, punctuation is preserved and punctuations_{in, out} are used
    )
    punctuations_out = '、。，"(),.:;¿?¡!\\'  # punctuation which usualy occurs outside words (important during phonemization)
    punctuations_in = "'-"  # punctuation which can occur inside a word, so whitespaces do not have to be present
    use_phonemes = False  # phonemes are valid only if True, tacotron uses phonemes instead of characters
    # all phonemes of IPA: 'iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻʘɓǀɗǃʄǂɠǁʛpbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟˈˌːˑʍwɥʜʢʡɕʑɺɧ ɚ˞ɫ'
    phonemes = "ɹɐpbtdkɡfvθðszʃʒhmnŋlrwjeəɪɒuːɛiaʌʊɑɜɔx "  # supported input phonemes, used if use_phonemes is True
    """
    ******************** PARAMETERS OF MODEL ********************
    """

    embedding_dimension = 512  # dimension of character embedding
    encoder_type = "simple"  # changes type of the encoder of the Tacotron 2 tacotron
    # one of: simple (single vanilla encoder for all languages without embedding),
    #         separate (distinct vanilla encoders for each language)
    #         shared (single vanilla encoder for all languages with lang. embedding injected to character embeddings)
    #         convolutional (single grouped fully convolutional encoder without embedding, each group corresponds to a language)
    #         generated (same as convolutional but with parameters generated by a meta-learning network)
    encoder_dimension = 512  # output dimension of the encoder
    encoder_blocks = 3  # number of convolutional block in vanilla encoder
    encoder_kernel_size = 5  # size of kernel of convolutional blocks in vanilla encoder
    generator_dim = 8  # used if encoder_type is 'generated', size of the 'language embedding' which is used by layers to generate weights
    generator_bottleneck_dim = 4  # used if encoder_type is 'generated', size of fully-connected layers which generate parameters for encoder layers
    prenet_dimension = 256  # size of pre-net layers
    prenet_layers = 2  # number of pre-net layers
    attention_type = "location_sensitive"  # Type of the attention mechanism.
    # one of: location_sensitive (Tacotron 2 vanilla),
    #         forward (undebugged, should allow just monotonous att.)
    #         forward_transition_agent (undebugged, fwd with explicit transition agent)
    attention_dimension = 128  #
    attention_kernel_size = 31  # kernel size of the convolution which extracts features from attention weights
    attention_location_dimension = 32  # size of the features extracted by a convolutional layer from attention weights
    decoder_dimension = 1024  # size of decoder RNNs
    decoder_regularization = (
        "dropout"  # regularization of decoder RNNs, one of: 'dropout', 'zoneout'
    )
    zoneout_hidden = (
        0.1  # used if decoder_regularization is 'zoneout', zoneout rate of LSTM h state
    )
    zoneout_cell = (
        0.1  # used if decoder_regularization is 'zoneout', zoneout rate of LSTM c state
    )
    dropout_hidden = (
        0.1  # used if decoder_regularization is 'dropout', dropout rate of LSTM output
    )
    postnet_dimension = 512  # size of post-net layers
    postnet_blocks = 5  # number of convolutional blocks in post-net
    postnet_kernel_size = 5  # kernel size of convolutions in post-net blocks
    dropout = 0.5  # dropout rate of convolutional block in the whole tacotron
    predict_linear = False  # if True, vanilla post-net is replaced by CBHG module which predicts linear spectrograms
    cbhg_bank_kernels = 8  # used if predict_linear is True
    cbhg_bank_dimension = 128  # used if predict_linear is True
    cbhg_projection_kernel_size = 3  # used if predict_linear is True
    cbhg_projection_dimension = 256  # used if predict_linear is True
    cbhg_highway_dimension = 128  # used if predict_linear is True
    cbhg_rnn_dim = 128  # used if predict_linear is True
    cbhg_dropout = 0.0  # used if predict_linear is True
    multi_speaker = False  # if True, multi-speaker tacotron is used, speaker embeddings are concatenated to encoder outputs
    multi_language = False  # if True, multi-lingual tacotron is used, language embeddings are concatenated to encoder outputs
    speaker_embedding_dimension = (
        32  # used if multi_speaker is True, size of the speaker embedding
    )
    language_embedding_dimension = (
        4  # used if multi_language is True, size of the language embedding
    )
    input_language_embedding = 4  # used if encoder_type is 'shared', language embedding of this size is concatenated to input char. embeddings
    reversal_classifier = False  # if True, adversarial classifier for predicting speakers from encoder outputs is used
    reversal_classifier_type = "reversal"  # one of: 'reversal' for a standard adversarial process with reverted gradients
    #           'cosine' for a cosine similarity-based adversarial process, does not converge at all
    reversal_classifier_dim = 256  # used if reversal_classifier is True and reversal_classifier_type id 'reversal'
    # size of the hidden layer of the adversarial classifer
    reversal_classifier_w = 1.0  # weight of the loss of the adversarial classifier (it is also reduced by number of mels, see TacotronLoss)
    stop_frames = 5  # number of frames at the end which are considered as "ending sequence" and stop token probability should be one
    speaker_number = 0  # do not set!
    language_number = 0  # do not set!
    """
    ******************** PARAMETERS OF AUDIO ********************
    """

    sample_rate = 22050  # sample rate of source .wavs, used while computing spectrograms, MFCCs, etc.
    num_fft = 1102  # number of frequency bins used during computation of spectrograms
    num_mels = 80  # number of mel bins used during computation of mel spectrograms
    num_mfcc = 13  # number of MFCCs, used just for MCD computation (during training)
    stft_window_ms = 50  # size in ms of the Hann window of short-time Fourier transform, used during spectrogram computation
    stft_shift_ms = (
        12.5  # shift of the window (or better said gap between windows) in ms
    )
    griffin_lim_iters = 60  # used if vocoding using Griffin-Lim algorithm (synthesize.py), greater value does not make much sense
    griffin_lim_power = 1.5  # power applied to spectrograms before using GL
    normalize_spectrogram = True  # if True, spectrograms are normalized before passing into the tacotron, a per-channel normalization is used
    # statistics (mean and variance) are computed from dataset at the start of training
    use_preemphasis = True  # if True, a preemphasis is applied to raw waveform before using them (spectrogram computation)
    preemphasis = 0.97  # amount of preemphasis, used if use_preemphasis is True

    @staticmethod
    def load_state_dict(d):
        for k, v in d.items():
            setattr(Params, k, v)

    @staticmethod
    def state_dict():
        members = [
            attr for attr in dir(Params)
            if not callable(getattr(Params, attr)) and not attr.startswith("__")
        ]
        return {k: Params.__dict__[k] for k in members}

    @staticmethod
    def load(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            params = json.load(f)
            Params.load_state_dict(params)

    @staticmethod
    def save(json_path):
        with open(json_path, "w", encoding="utf-8") as f:
            d = Params.state_dict()
            json.dump(d, f, indent=4)

    @staticmethod
    def symbols_count():
        symbols_count = len(Params.characters)
        if Params.use_phonemes:
            symbols_count = len(Params.phonemes)
        if Params.use_punctuation:
            symbols_count += len(Params.punctuations_out) + len(
                Params.punctuations_in)
        return symbols_count
