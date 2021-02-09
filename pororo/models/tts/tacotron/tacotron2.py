from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential
from torch.nn import functional as F

from pororo.models.tts.tacotron.attention import LocationSensitiveAttention
from pororo.models.tts.tacotron.encoder import GeneratedConvolutionalEncoder
from pororo.models.tts.tacotron.layers import (
    ConvBlock,
    DropoutLSTMCell,
    ZoneoutLSTMCell,
)
from pororo.models.tts.tacotron.params import Params as hp
from pororo.models.tts.utils import *


class Prenet(torch.nn.Module):
    """Decoder pre-net module.

    Details:
        stack of 2 linear layers with dropout which is enabled even during inference (output variation)
        should act as a bottleneck for the attention

    Arguments:
        input_dim -- size of the input (supposed the number of frame mels)
        output_dim -- size of the output
        num_layers -- number of the linear layers (at least one)
        dropout -- dropout rate to be aplied after each layer (even during inference)
    """

    def __init__(self, input_dim, output_dim, num_layers, dropout):
        super(Prenet, self).__init__()
        assert num_layers > 0, "There must be at least one layer in the pre-net."
        self._dropout_rate = dropout
        self._activation = ReLU()
        layers = [Linear(input_dim, output_dim)] + [
            Linear(output_dim, output_dim) for _ in range(num_layers - 1)
        ]
        self._layers = ModuleList(layers)

    def _layer_pass(self, x, layer):
        x = layer(x)
        x = self._activation(x)
        x = F.dropout(x, p=self._dropout_rate, training=True)
        return x

    def forward(self, x):
        for layer in self._layers:
            x = self._layer_pass(x, layer)
        return x


class Postnet(torch.nn.Module):
    """Post-net module for output spectrogram enhancement.

    Details:
        stack of 5 conv. layers 5 × 1 with BN and tanh (except last), dropout

    Arguments:
        input_dimension -- size of the input and output (supposed the number of frame mels)
        postnet_dimension -- size of the internal convolutional blocks
        num_blocks -- number of the convolutional blocks (at least one)
        kernel_size -- kernel size of the encoder's convolutional blocks
        dropout -- dropout rate to be aplied after each convolutional block
    """

    def __init__(self, input_dimension, postnet_dimension, num_blocks,
                 kernel_size, dropout):
        super(Postnet, self).__init__()
        assert (
            num_blocks > 1
        ), "There must be at least two convolutional blocks in the post-net."
        self._convs = Sequential(
            ConvBlock(input_dimension, postnet_dimension, kernel_size, dropout,
                      "tanh"), *[
                          ConvBlock(
                              postnet_dimension,
                              postnet_dimension,
                              kernel_size,
                              dropout,
                              "tanh",
                          ) for _ in range(num_blocks - 2)
                      ],
            ConvBlock(
                postnet_dimension,
                input_dimension,
                kernel_size,
                dropout,
                "identity",
            ))

    def forward(self, x, x_lengths):
        residual = x
        x = self._convs(x)
        x += residual
        return x


class Decoder(torch.nn.Module):
    """Tacotron 2 decoder with queries produced by the first RNN layer and output produced by the second RNN.

    Decoder:
        stack of 2 uni-directional LSTM layers with 1024 units
        first LSTM is used to query attention mechanism
        input of the first LSTM is previous prediction (pre-net output) and previous context vector
        second LSTM acts as a generator
        input of the second LSTM is current context vector and output of the first LSTM
        output is passed through stop token layer, frame prediction layer and pre-net

    Arguments:
        output_dim -- size of the predicted frame, i.e. number of mels
        decoder_dim -- size of the generator output (and also of all the LSTMs used in the decoder)
        attention -- instance of the location-sensitive attention module
        generator_rnn -- instance of generator RNN
        attention_rnn -- instance of attention RNN
        context_dim -- size of the context vector produced by the given attention
        prenet -- instance of the pre-net module
        prenet_dim -- output dimension of the pre-net
        max_frames -- maximal number of the predicted frames
    """

    def __init__(
        self,
        output_dim,
        decoder_dim,
        attention,
        generator_rnn,
        attention_rnn,
        context_dim,
        prenet,
        prenet_dim,
        max_frames,
    ):
        super(Decoder, self).__init__()
        self._prenet = prenet
        self._attention = attention
        self._output_dim = output_dim
        self._decoder_dim = decoder_dim
        self._max_frames = max_frames
        self._attention_lstm = attention_rnn
        self._generator_lstm = generator_rnn
        self._frame_prediction = Linear(context_dim + decoder_dim, output_dim)
        self._stop_prediction = Linear(context_dim + decoder_dim, 1)

        self._speaker_embedding, self._language_embedding = None, None

        if hp.multi_speaker and hp.speaker_embedding_dimension > 0:
            self._speaker_embedding = self._get_embedding(
                hp.speaker_embedding_dimension, hp.speaker_number)
        if hp.multi_language and hp.language_embedding_dimension > 0:
            self._language_embedding = self._get_embedding(
                hp.language_embedding_dimension, len(hp.languages))

    def _get_embedding(self, embedding_dimension, size=None):
        embedding = Embedding(size, embedding_dimension)
        torch.nn.init.xavier_uniform_(embedding.weight)
        return embedding

    def _target_init(self, target, batch_size):
        """Prepend target spectrogram with a zero frame and pass it through pre-net."""
        # the F.pad function has some issues: https://github.com/pytorch/pytorch/issues/13058
        first_frame = torch.zeros(
            batch_size,
            self._output_dim,
            device=target.device,
        ).unsqueeze(1)
        target = target.transpose(1, 2)  # [B, F, N_MEL]
        target = torch.cat((first_frame, target), dim=1)
        target = self._prenet(target)
        return target

    def _decoder_init(self, batch_size, device):
        """Initialize hidden and cell state of the deocder's RNNs."""
        h_att = torch.zeros(batch_size, self._decoder_dim, device=device)
        c_att = torch.zeros(batch_size, self._decoder_dim, device=device)
        h_gen = torch.zeros(batch_size, self._decoder_dim, device=device)
        c_gen = torch.zeros(batch_size, self._decoder_dim, device=device)
        return h_att, c_att, h_gen, c_gen

    def _add_conditional_embedding(self, encoded, layer, condition):
        """Compute speaker (lang.) embedding and concat it to the encoder output."""
        embedded = layer(encoded if condition is None else condition)
        return torch.cat((encoded, embedded), dim=-1)

    def _decode(self, encoded_input, mask, target, teacher_forcing_ratio,
                speaker, language):
        """Perform decoding of the encoded input sequence."""

        batch_size = encoded_input.size(0)
        max_length = encoded_input.size(1)
        inference = target is None
        max_frames = self._max_frames if inference else target.size(2)
        input_device = encoded_input.device

        # obtain speaker and language embeddings (or a dummy tensor)
        if hp.multi_speaker and self._speaker_embedding is not None:
            encoded_input = self._add_conditional_embedding(
                encoded_input, self._speaker_embedding, speaker)
        if hp.multi_language and self._language_embedding is not None:
            encoded_input = self._add_conditional_embedding(
                encoded_input, self._language_embedding, language)

        # attention and decoder states initialization
        context = self._attention.reset(
            encoded_input,
            batch_size,
            max_length,
            input_device,
        )
        h_att, c_att, h_gen, c_gen = self._decoder_init(
            batch_size,
            input_device,
        )

        # prepare some inference or train specific variables (teacher forcing, max. predicted length)
        frame = torch.zeros(batch_size, self._output_dim, device=input_device)
        if not inference:
            target = self._target_init(target, batch_size)
            teacher = torch.rand(
                [max_frames], device=input_device) > (1 - teacher_forcing_ratio)

        # tensors for storing output
        spectrogram = torch.zeros(
            batch_size,
            max_frames,
            self._output_dim,
            device=input_device,
        )
        alignments = torch.zeros(
            batch_size,
            max_frames,
            max_length,
            device=input_device,
        )
        stop_tokens = torch.zeros(
            batch_size,
            max_frames,
            1,
            device=input_device,
        )

        # decoding loop
        stop_frames = -1
        for i in range(max_frames):
            prev_frame = (self._prenet(frame)
                          if inference or not teacher[i] else target[:, i])

            # run decoder attention and RNNs
            attention_input = torch.cat((prev_frame, context), dim=1)
            h_att, c_att = self._attention_lstm(attention_input, h_att, c_att)
            context, weights = self._attention(
                h_att,
                encoded_input,
                mask,
                prev_frame,
            )
            generator_input = torch.cat((h_att, context), dim=1)
            h_gen, c_gen = self._generator_lstm(generator_input, h_gen, c_gen)

            # predict frame and stop token
            proto_output = torch.cat((h_gen, context), dim=1)
            frame = self._frame_prediction(proto_output)
            stop_logits = self._stop_prediction(proto_output)

            # store outputs
            spectrogram[:, i] = frame
            alignments[:, i] = weights
            stop_tokens[:, i] = stop_logits

            # stop decoding if predicted (just during inference)
            if inference and torch.sigmoid(stop_logits).ge(0.5):
                if stop_frames == -1:
                    stop_frames = hp.stop_frames
                    continue
                stop_frames -= 1
                if stop_frames == 0:
                    return (
                        spectrogram[:, :i + 1],
                        stop_tokens[:, :i + 1].squeeze(2),
                        alignments[:, :i + 1],
                    )

        return spectrogram, stop_tokens.squeeze(2), alignments

    def forward(
        self,
        encoded_input,
        encoded_lenghts,
        target,
        teacher_forcing_ratio,
        speaker,
        language,
    ):
        ml = encoded_input.size(1)
        mask = lengths_to_mask(encoded_lenghts, max_length=ml)
        return self._decode(
            encoded_input,
            mask,
            target,
            teacher_forcing_ratio,
            speaker,
            language,
        )

    def inference(self, encoded_input, speaker, language):
        mask = lengths_to_mask(torch.LongTensor([encoded_input.size(1)]))
        spectrogram, _, _ = self._decode(
            encoded_input,
            mask,
            None,
            0.0,
            speaker,
            language,
        )
        return spectrogram


class Tacotron(torch.nn.Module):
    """
    Tacotron 2:
        characters as learned embedding
        encoder, attention, decoder which predicts frames of mel spectrogram
        the predicted mel spectrogram is passed through post-net which
          predicts a residual to add to the prediction
        minimize MSE from before and after the post-net to aid convergence
    """

    def __init__(self):
        super(Tacotron, self).__init__()

        # Encoder embedding
        other_symbols = 3  # PAD, EOS, UNK
        self._embedding = Embedding(
            hp.symbols_count() + other_symbols,
            hp.embedding_dimension,
            padding_idx=0,
        )
        torch.nn.init.xavier_uniform_(self._embedding.weight)

        # Encoder transforming graphmenes or phonemes into abstract input representation
        self._encoder = self._get_encoder()

        # Prenet for transformation of previous predicted frame
        self._prenet = Prenet(
            hp.num_mels,
            hp.prenet_dimension,
            hp.prenet_layers,
            hp.dropout,
        )

        # Speaker and language embeddings make decoder bigger
        decoder_input_dimension = hp.encoder_dimension
        if hp.multi_speaker:
            decoder_input_dimension += hp.speaker_embedding_dimension
        if hp.multi_language:
            decoder_input_dimension += hp.language_embedding_dimension

        # Decoder attention layer
        self._attention = self._get_attention(decoder_input_dimension)

        # Instantiate decoder RNN layers
        gen_cell_dimension = decoder_input_dimension + hp.decoder_dimension
        att_cell_dimension = decoder_input_dimension + hp.prenet_dimension
        if hp.decoder_regularization == "zoneout":
            generator_rnn = ZoneoutLSTMCell(
                gen_cell_dimension,
                hp.decoder_dimension,
                hp.zoneout_hidden,
                hp.zoneout_cell,
            )
            attention_rnn = ZoneoutLSTMCell(
                att_cell_dimension,
                hp.decoder_dimension,
                hp.zoneout_hidden,
                hp.zoneout_cell,
            )
        else:
            generator_rnn = DropoutLSTMCell(
                gen_cell_dimension,
                hp.decoder_dimension,
                hp.dropout_hidden,
            )
            attention_rnn = DropoutLSTMCell(
                att_cell_dimension,
                hp.decoder_dimension,
                hp.dropout_hidden,
            )

        # Decoder which controls attention and produces mel frames and stop tokens
        self._decoder = Decoder(
            hp.num_mels,
            hp.decoder_dimension,
            self._attention,
            generator_rnn,
            attention_rnn,
            decoder_input_dimension,
            self._prenet,
            hp.prenet_dimension,
            hp.max_output_length,
        )

        # Postnet transforming predicted mel frames (residual mel or linear frames)
        self._postnet = self._get_postnet()

    def _get_encoder(self):
        args = (
            hp.embedding_dimension,
            hp.encoder_dimension,
            hp.encoder_blocks,
            hp.encoder_kernel_size,
            hp.dropout,
        )
        ln = 1 if not hp.multi_language else hp.language_number
        return GeneratedConvolutionalEncoder(
            hp.embedding_dimension,
            hp.encoder_dimension,
            0.05,
            hp.generator_dim,
            hp.generator_bottleneck_dim,
            groups=ln,
        )

    def _get_attention(self, memory_dimension):
        args = (hp.attention_dimension, hp.decoder_dimension, memory_dimension)
        return LocationSensitiveAttention(
            hp.attention_kernel_size,
            hp.attention_location_dimension,
            False,
            *args,
        )

    def _get_postnet(self):
        return Postnet(
            hp.num_mels,
            hp.postnet_dimension,
            hp.postnet_blocks,
            hp.postnet_kernel_size,
            hp.dropout,
        )

    def forward(
        self,
        text,
        text_length,
        target,
        target_length,
        speakers,
        languages,
        teacher_forcing_ratio=0.0,
    ):
        # enlarge speakers and languages to match sentence length if needed
        if speakers is not None and speakers.dim() == 1:
            speakers = speakers.unsqueeze(1).expand((-1, text.size(1)))
        if languages is not None and languages.dim() == 1:
            languages = languages.unsqueeze(1).expand((-1, text.size(1)))

        # encode input
        embedded = self._embedding(text)
        encoded = self._encoder(embedded, text_length, languages)
        encoder_output = encoded

        # predict language as an adversarial task if needed
        speaker_prediction = (self._reversal_classifier(encoded)
                              if hp.reversal_classifier else None)

        # decode
        if languages is not None and languages.dim() == 3:
            languages = torch.argmax(
                languages,
                dim=2,
            )  # convert one-hot into indices
        decoded = self._decoder(
            encoded,
            text_length,
            target,
            teacher_forcing_ratio,
            speakers,
            languages,
        )
        prediction, stop_token, alignment = decoded
        pre_prediction = prediction.transpose(1, 2)
        post_prediction = self._postnet(pre_prediction, target_length)

        # mask output paddings
        target_mask = lengths_to_mask(target_length, target.size(2))
        stop_token.masked_fill_(~target_mask, 1000)
        target_mask = target_mask.unsqueeze(1).float()
        pre_prediction = pre_prediction * target_mask
        post_prediction = post_prediction * target_mask

        return (
            post_prediction,
            pre_prediction,
            stop_token,
            alignment,
            speaker_prediction,
            encoder_output,
        )

    def inference(self, text, speaker=None, language=None):
        # pretend having a batch of size 1
        text.unsqueeze_(0)

        if speaker is not None and speaker.dim() == 1:
            speaker = speaker.unsqueeze(1).expand((-1, text.size(1)))
        if language is not None and language.dim() == 1:
            language = language.unsqueeze(1).expand((-1, text.size(1)))

        # encode input
        embedded = self._embedding(text)
        encoded = self._encoder(
            embedded,
            torch.LongTensor([text.size(1)]),
            language,
        )

        # decode with respect to speaker and language embeddings
        if language is not None and language.dim() == 3:
            language = torch.argmax(
                language,
                dim=2,
            )  # convert one-hot into indices
        prediction = self._decoder.inference(encoded, speaker, language)

        # post process generated spectrogram
        prediction = prediction.transpose(1, 2)
        post_prediction = self._postnet(
            prediction,
            torch.LongTensor([prediction.size(2)]),
        )
        return post_prediction.squeeze(0)
