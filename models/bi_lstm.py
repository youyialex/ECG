import torch.nn as nn


class RNN1d(nn.Sequential):
    def __init__(self, num_classes, input_channels=12, hidden_dim=256, num_layers=2, bidirectional=False):
        super(RNN1d, self).__init__()
        self.lstm = nn.LSTM(input_size=input_channels, hidden_size=hidden_dim, num_layers=num_layers,
                            bidirectional=bidirectional)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        if not bidirectional:
            self.fc = nn.Linear(256, num_classes)
        else:
            self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        #input=(batch,length,channels) changed to
        #input = torch.randn(5, 3, 10) # 
        # 输入数据由3个句子组成，每个句子由5个单词组成，单词向量长度为10
        x = x.transpose(1, 2)
        x = x.transpose(0, 1)
        #input=(channels,batch,length)
        output = self.lstm(x)
        output = output[0].transpose(0, 1)
        output = output.transpose(1, 2)
        output = self.avgpool(output).squeeze(-1)
        output = self.fc(output)
        return output


def lstm(**kwargs):
    return RNN1d(bidirectional=False, **kwargs)


def lstm_bidir(**kwargs):
    return RNN1d(bidirectional=True, **kwargs)


