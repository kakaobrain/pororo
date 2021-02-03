# Copyright (c) Hyeon Kyu Lee and Kakao Brain. All Rights Reserved

import numpy as np
import torch
import torch.nn as nn


def same_padding(kernel):
    pad_val = (kernel - 1) / 2

    if kernel % 2 == 0:
        out = (int(pad_val - 0.5), int(pad_val + 0.5))
    else:
        out = int(pad_val)

    return out


class VoiceActivityDetection(object):
    """
    Voice activity detection (VAD), also known as speech activity detection or speech detection,
    is the detection of the presence or absence of human speech, used in speech processing.

    Args: model_path, device
        model_path: path of vad model
        device: 'cuda' or 'cpu'
    """

    def __init__(self, model_path: str, device: str):
        import librosa

        self.librosa = librosa

        self.sample_rate = 16000
        self.n_mfcc = 5
        self.n_mels = 40
        self.device = device

        self.model = ConvVADModel()
        self.model.load_state_dict(torch.load(model_path, map_location=device))

        self.model.to(device).eval()

    def extract_features(
        self,
        signal,
        size: int = 512,
        step: int = 16,
    ):
        # Mel Frequency Cepstral Coefficents
        mfcc = self.librosa.feature.mfcc(
            y=signal,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=size,
            hop_length=step,
        )
        mfcc_delta = self.librosa.feature.delta(mfcc)
        mfcc_delta2 = self.librosa.feature.delta(mfcc, order=2)

        # Root Mean Square Energy
        melspectrogram = self.librosa.feature.melspectrogram(
            y=signal,
            n_mels=self.n_mels,
            sr=self.sample_rate,
            n_fft=size,
            hop_length=step,
        )
        rmse = self.librosa.feature.rms(
            S=melspectrogram,
            frame_length=self.n_mels * 2 - 1,
            hop_length=step,
        )

        mfcc = np.asarray(mfcc)
        mfcc_delta = np.asarray(mfcc_delta)
        mfcc_delta2 = np.asarray(mfcc_delta2)
        rmse = np.asarray(rmse)

        features = np.concatenate((mfcc, mfcc_delta, mfcc_delta2, rmse), axis=0)
        features = np.transpose(features)

        return features

    def smooth_predictions_v1(self, label):
        smoothed_label = list()

        # Smooth with 3 consecutive windows
        for i in range(2, len(label), 3):
            cur_pred = label[i]
            if cur_pred == label[i - 1] == label[i - 2]:
                smoothed_label.extend([cur_pred, cur_pred, cur_pred])
            else:
                if len(smoothed_label) > 0:
                    smoothed_label.extend([
                        smoothed_label[-1], smoothed_label[-1],
                        smoothed_label[-1]
                    ])
                else:
                    smoothed_label.extend([0, 0, 0])

        n = 0
        while n < len(smoothed_label):
            cur_pred = smoothed_label[n]
            if cur_pred == 1:
                if n > 0:
                    smoothed_label[n - 1] = 1
                if n < len(smoothed_label) - 1:
                    smoothed_label[n + 1] = 1
                n += 2
            else:
                n += 1

        for idx in range(len(label) - len(smoothed_label)):
            smoothed_label.append(smoothed_label[-1])

        return smoothed_label

    def smooth_predictions_v2(self, label):
        smoothed_label = list()
        # Smooth with 3 consecutive windows
        for i in range(2, len(label)):
            cur_pred = label[i]
            if cur_pred == label[i - 1] == label[i - 2]:
                smoothed_label.append(cur_pred)
            else:
                if len(smoothed_label) > 0:
                    smoothed_label.append(smoothed_label[-1])
                else:
                    smoothed_label.append(0)

        n = 0
        while n < len(smoothed_label):
            cur_pred = smoothed_label[n]
            if cur_pred == 1:
                if n > 0:
                    smoothed_label[n - 1] = 1
                if n < len(smoothed_label) - 1:
                    smoothed_label[n + 1] = 1
                n += 2
            else:
                n += 1

        for _ in range(len(label) - len(smoothed_label)):
            smoothed_label.append(smoothed_label[-1])

        return smoothed_label

    def get_speech_intervals(self, data, label):

        def get_speech_interval(labels):
            seguence_length = 1024
            speech_interval = [[0, 0]]
            pre_label = 0

            for idx, label in enumerate(labels):

                if label:
                    if pre_label == 1:
                        speech_interval[-1][1] = (idx + 1) * seguence_length
                    else:
                        speech_interval.append([
                            idx * seguence_length, (idx + 1) * seguence_length
                        ])

                pre_label = label

            return speech_interval[1:]

        speech_intervals = list()
        interval = get_speech_interval(label)

        for start, end in interval:
            speech_intervals.append(data[start:end])

        return speech_intervals

    def __call__(self, signal: np.ndarray, sample_rate: int = 16000):
        seguence_signal = list()

        self.sample_rate = sample_rate
        start_pointer = 0
        end_pointer = 1024

        while end_pointer < len(signal):
            seguence_signal.append(signal[start_pointer:end_pointer])

            start_pointer = end_pointer
            end_pointer += 1024

        feature = [self.extract_features(signal) for signal in seguence_signal]

        feature = np.array(feature)
        feature = np.expand_dims(feature, 1)
        x_tensor = torch.from_numpy(feature).float().to(self.device)

        output = self.model(x_tensor)
        predicted = torch.max(output.data, 1)[1]

        predict_label = predicted.to(torch.device("cpu")).detach().numpy()

        predict_label = self.smooth_predictions_v2(predict_label)
        predict_label = self.smooth_predictions_v1(predict_label)

        return self.get_speech_intervals(signal, predict_label)


class ResnetBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_kernels1: tuple,
        num_kernels2: int,
    ):
        super(ResnetBlock, self).__init__()

        padding = same_padding(num_kernels1[0])
        self.zero_pad = nn.ZeroPad2d((
            0,
            0,
            padding[0],
            padding[1],
        ))
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            (num_kernels1[0], num_kernels2),
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            num_kernels1[1],
            padding=same_padding(num_kernels1[1]),
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels,
            num_kernels1[2],
            padding=same_padding(num_kernels1[2]),
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, (1, num_kernels2))
        self.bn_shortcut = nn.BatchNorm2d(out_channels)
        self.out_block = nn.ReLU()

    def forward(self, inputs):
        x = self.zero_pad(inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        shortcut = self.shortcut(inputs)
        shortcut = self.bn_shortcut(shortcut)
        x = torch.add(x, shortcut)
        out_block = self.out_block(x)

        return out_block


class ConvVADModel(nn.Module):

    def __init__(self):
        super(ConvVADModel, self).__init__()

        self.block1 = ResnetBlock(
            in_channels=1,
            out_channels=32,
            num_kernels1=(8, 5, 3),
            num_kernels2=16,
        )
        self.block2 = ResnetBlock(
            in_channels=32,
            out_channels=64,
            num_kernels1=(8, 5, 3),
            num_kernels2=1,
        )
        self.block3 = ResnetBlock(
            in_channels=64,
            out_channels=128,
            num_kernels1=(8, 5, 3),
            num_kernels2=1,
        )
        self.block4 = ResnetBlock(
            in_channels=128,
            out_channels=128,
            num_kernels1=(8, 5, 3),
            num_kernels2=1,
        )

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(128 * 65, 2048)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(2048, 2048)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(2048, 2)

    def forward(self, inputs):
        out_block1 = self.block1(inputs)
        out_block2 = self.block2(out_block1)
        out_block3 = self.block3(out_block2)
        out_block4 = self.block4(out_block3)

        x = self.flat(out_block4)
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)

        return output
