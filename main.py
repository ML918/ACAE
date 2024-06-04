import argparse
import torch
from torch import nn
from data_loader_ import get_loader_dataset
import utils_


class Bottleneck(nn.Module):
    def __init__(self, C_in, C_out, expansion, isEnc=True, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.expansion = expansion
        self.Conv1 = nn.Conv1d(C_in, C_out, kernel_size=(1, ), stride=(1, ), padding=0, bias=False)
        self.BN1 = nn.BatchNorm1d(C_out)
        if isEnc:
            self.Conv2 = nn.Conv1d(C_out, C_out, kernel_size=(3,), stride=(stride,), padding=1, bias=False)
        else:
            self.Conv2 = nn.ConvTranspose1d(C_out, C_out, kernel_size=(4,), stride=(stride,), padding=(1,), bias=False)
        self.BN2 = nn.BatchNorm1d(C_out)
        self.Conv3 = nn.Conv1d(C_out, int(C_out * self.expansion), kernel_size=(1,), stride=(1,), padding=0, bias=False)
        self.BN3 = nn.BatchNorm1d(int(C_out * self.expansion))

    def forward(self, X):
        identity = X
        if self.downsample is not None:
            identity = self.downsample(X)
        out = torch.tanh(self.BN1(self.Conv1(X)))
        out = torch.tanh(self.BN2(self.Conv2(out)))
        out = torch.tanh(self.BN3(self.Conv3(out)) + identity)
        return out


class ResNet_Encoder(nn.Module):
    def __init__(self, blocks_num, n_feature, embedding_size, num_hidden):
        super(ResNet_Encoder, self).__init__()
        self.in_channel = embedding_size
        self.expansion = 4
        self.emb1 = nn.Conv1d(n_feature, embedding_size, kernel_size=(1,), padding=0, bias=False)
        self.BN1 = nn.BatchNorm1d(embedding_size)
        self.emb2 = nn.Conv1d(embedding_size, embedding_size, kernel_size=(3,), padding=1, bias=False)
        self.BN2 = nn.BatchNorm1d(embedding_size)
        self.emb3 = nn.Conv1d(embedding_size, embedding_size, kernel_size=(7,), stride=(2,), padding=3, bias=False)
        self.BN3 = nn.BatchNorm1d(embedding_size)
        self.emb4 = nn.MaxPool1d(kernel_size=(3,), stride=(2,), padding=1)

        self.layer1 = self._add_layer(64, blocks_num[0], stride=2)
        self.layer2 = self._add_layer(128, blocks_num[1], stride=2)
        self.AvgPool = nn.AdaptiveAvgPool1d((1, ))
        self.out = nn.Linear(128 * self.expansion, num_hidden)

    def _add_layer(self, channel, block_num, stride):
        downsample = nn.Sequential(
            nn.Conv1d(self.in_channel, channel * self.expansion, kernel_size=(1,), stride=(stride,), bias=False),
            nn.BatchNorm1d(channel * self.expansion))

        layers = []
        layers.append(Bottleneck(self.in_channel, channel, self.expansion, True, stride, downsample))
        self.in_channel = channel * self.expansion
        for _ in range(1, block_num):
            layers.append(Bottleneck(self.in_channel, channel, self.expansion))
        return nn.Sequential(*layers)

    def forward(self, X):
        X = X.permute(0, 2, 1)
        out = torch.tanh(self.BN1(self.emb1(X)))
        out = torch.tanh(self.BN2(self.emb2(out)))
        out = torch.tanh(self.BN3(self.emb3(out)))
        out = self.emb4(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = torch.tanh(self.out(self.AvgPool(out).squeeze(-1)))
        return out


class ResNet_Decoder(nn.Module):
    def __init__(self, blocks_num, n_feature, num_hidden, win_size):
        super(ResNet_Decoder, self).__init__()
        self.in_channel = 512
        self.expansion = 0.25

        self.input1 = nn.Linear(num_hidden, 512)
        self.input2 = nn.Conv1d(1, int(win_size / 4), kernel_size=(1, ), stride=(1, ), padding=0, bias=False)
        self.BN1 = nn.BatchNorm1d(512)
        self.layer1 = self._add_layer(512, blocks_num[0], stride=2)
        self.layer2 = self._add_layer(256, blocks_num[1], stride=2)

        self.out = nn.Linear(64, n_feature)

    def _add_layer(self, channel, block_num, stride):
        upsample = nn.Sequential(
            nn.ConvTranspose1d(self.in_channel, int(channel * self.expansion),
                               kernel_size=(2,), stride=(stride,), bias=False),
            nn.BatchNorm1d(int(channel * self.expansion)))

        layers = []
        layers.append(Bottleneck(self.in_channel, channel, self.expansion, False, stride, upsample))
        self.in_channel = int(channel * self.expansion)
        for _ in range(1, block_num):
            layers.append(Bottleneck(self.in_channel, channel, self.expansion))
        return nn.Sequential(*layers)

    def forward(self, H):
        out = torch.tanh(self.input1(H)).unsqueeze(-1).permute(0, 2, 1)
        out = torch.tanh(self.BN1(self.input2(out).permute(0, 2, 1)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.out(out.permute(0, 2, 1))
        return out


def mask_(X, mask_rate, device):
    mask = torch.ones(X.shape, device=device)
    for i in range(X.shape[0]):
        r = torch.randperm(X.shape[1])
        for j in range(int(X.shape[1] * mask_rate)):
            mask[i][r[j]][:] = 0
    mask_X = X * mask
    return mask_X


class Encoder(nn.Module):
    def __init__(self, n_features, embedding_size, num_hiddens, device):
        super(Encoder, self).__init__()
        self.device = device
        self.projection = nn.Linear(n_features, embedding_size)
        self.encoder = ResNet_Encoder([3, 3], embedding_size, embedding_size, num_hiddens)

    def forward(self, X, con=False):
        if con:
            emb_X = self.projection(X)
            mask_1 = mask_(emb_X, 0.05, self.device)
            mask_2 = mask_(emb_X, 0.15, self.device)
            mask_3 = mask_(emb_X, 0.3, self.device)
            mask_4 = mask_(emb_X, 0.5, self.device)
            real_H = self.encoder(emb_X)
            mask_H1 = self.encoder(mask_1)
            mask_H2 = self.encoder(mask_2)
            mask_H3 = self.encoder(mask_3)
            mask_H4 = self.encoder(mask_4)
            return real_H, mask_H1, mask_H2, mask_H3, mask_H4
        else:
            emb_X = self.projection(X)
            real_H = self.encoder(emb_X)
            return real_H


class Discriminator(nn.Module):
    def __init__(self, num_hiddens, dropout=0.5):
        super(Discriminator, self).__init__()
        self.out1 = nn.Linear(num_hiddens * 2, num_hiddens * 2)
        self.out2 = nn.Linear(num_hiddens * 2, num_hiddens)
        self.out3 = nn.Linear(num_hiddens, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H, H_):
        X = torch.cat((H, H_), dim=1)
        X = self.dropout(torch.relu(self.out1(X)))
        X = self.dropout(torch.relu(self.out2(X)))
        X = torch.sigmoid(self.out3(X))
        return X


def train(n_features, embedding_size, num_hiddens, win_size, n_fake, dropout, device, num_epochs, lr, weight_decay,
          patience, data_name, model_path, data_train, data_valid, data_test, lambda1, lambda2):
    encoder = Encoder(n_features, embedding_size, num_hiddens, device)
    decoder = ResNet_Decoder([3, 3], n_features, num_hiddens, win_size)
    discriminator = Discriminator(num_hiddens, dropout)
    utils_.xavier_initialize(encoder)
    utils_.xavier_initialize(decoder)
    encoder.to(device)
    decoder.to(device)
    discriminator.to(device)

    optim_ae = torch.optim.Adam([{'params': encoder.parameters()},
                                 {'params': decoder.parameters()}], lr=lr, weight_decay=weight_decay)
    optim_dis = torch.optim.Adam(discriminator.parameters(), lr=lr, weight_decay=weight_decay)
    optim_gen = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = utils_.EarlyStopping(patience, data_name, model_path)
    loss = torch.nn.MSELoss(reduction='mean')

    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        discriminator.train()
        metric = utils_.Accumulator(3)
        for X in data_train:
            X = X.to(device)
            b_size = X.shape[0]

            real_H, mask_H1, mask_H2, mask_H3, mask_H4 = encoder(X, con=True)
            fake_list = []
            for i in range(n_fake):
                idx = torch.randperm(b_size)
                fake_list.append(real_H[idx, :])
                idx = torch.randperm(b_size)
                fake_list.append(mask_H1[idx, :])
                idx = torch.randperm(b_size)
                fake_list.append(mask_H2[idx, :])
                idx = torch.randperm(b_size)
                fake_list.append(mask_H3[idx, :])
                idx = torch.randperm(b_size)
                fake_list.append(mask_H4[idx, :])

            alpha = torch.rand((4 * b_size, 1), device=device)
            beta = torch.rand((n_fake * 5 * b_size, 1), device=device)
            gamma = torch.cat((alpha, beta), dim=0)

            Mix_H1 = real_H.repeat(n_fake * 5 + 4, 1)
            mask_H = torch.cat((mask_H1, mask_H2, mask_H3, mask_H4), dim=0)
            fake_H = torch.cat(fake_list, dim=0)
            Mix_H2 = torch.cat((mask_H, fake_H), dim=0)
            Mix_H = gamma * Mix_H1 + (1 - gamma) * Mix_H2

            label1 = torch.cat((torch.ones((4 * b_size, 1), device=device), alpha), dim=1)
            label2 = torch.cat((torch.zeros((n_fake * 5 * b_size, 1), device=device), beta), dim=1)
            label = torch.cat((label1, label2), dim=0)

            pre = discriminator(Mix_H.detach(), Mix_H1.detach())
            l_dis = lambda1 * loss(pre, label)
            optim_dis.zero_grad()
            l_dis.backward()
            utils_.grad_clipping(discriminator, 1)
            optim_dis.step()

            label1 = torch.ones((4 * b_size, 2), device=device)
            label = torch.cat((label1, label2), dim=0)
            pre = discriminator(Mix_H, Mix_H1)
            l_gen = lambda2 * loss(pre, label)
            optim_gen.zero_grad()
            l_gen.backward()
            utils_.grad_clipping(encoder, 1)
            optim_gen.step()

            H = encoder(X)
            X_hat = decoder(H)
            l_rec = loss(X_hat, X)
            optim_ae.zero_grad()
            l_rec.backward()
            utils_.grad_clipping(encoder, 1)
            utils_.grad_clipping(decoder, 1)
            optim_ae.step()
            with torch.no_grad():
                metric.add(l_rec, l_dis, l_gen)

        with torch.no_grad():
            encoder.eval()
            decoder.eval()
            valid_loss = 0.0
            for X in data_valid:
                X = X.to(device)
                H = encoder(X)
                X_hat = decoder(H)
                valid_loss += loss(X_hat, X)
            print(f"epochs: {epoch + 1}")
            print(f"T: {metric[0]} V: {valid_loss} D: {metric[1]} G: {metric[2]}")
            early_stopping(valid_loss, encoder, decoder)
            if early_stopping.early_stop:
                print("EarlyStopping！")
                break

    encoder.load_state_dict(torch.load(model_path + data_name + '_encoder.pth'))
    decoder.load_state_dict(torch.load(model_path + data_name + '_decoder.pth'))

    encoder.eval()
    decoder.eval()
    test_loss = torch.nn.MSELoss(reduction='none')
    label, score = [], []
    for data in data_test:
        X, y = data[0], data[1]
        X = X.to(device)
        H = encoder(X)
        X_hat = decoder(H)
        l = test_loss(X_hat, X)
        l = torch.mean(l, dim=-1, keepdim=False)
        y = (torch.reshape(y, (-1, 1))).squeeze(dim=-1)
        l = (torch.reshape(l, (-1, 1))).squeeze(dim=-1)
        label.append(y.cpu().detach())
        score.append(l.cpu().detach())
    label = torch.cat(label, dim=0).numpy()
    score = torch.cat(score, dim=0).numpy()
    return label, score


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='data name', type=str, default='MSL')
    parser.add_argument('--seed', help='random seed', type=int, default=3407)
    parser.add_argument('--device', help='device', type=int, default=0)
    parser.add_argument('--data_path', help='data path', type=str, default="data/")
    parser.add_argument('--model_path', help='model save path', type=str, default='model/')
    parser.add_argument('--score_path', help='score save path', type=str, default="score/")
    args = parser.parse_args()

    utils_.setup_seed(args.seed)
    device = utils_.try_gpu(args.device)
    data_name = args.data
    data_path, model_path, score_path = args.data_path, args.model_path, args.score_path
    utils_.creat_path(model_path)
    utils_.creat_path(score_path)

    batch_size, win_size, step = 128, 128, 8
    num_hiddens, embedding_size, dropout = 128, 128, 0.5
    num_epochs, patience, lr, weight_decay = 200, 5, 1e-4, 1e-4
    lambda1, lambda2 = 1, 1
    if data_name == 'MSL':
        n_fake = 4
    elif data_name == 'PSM':
        n_fake = 16
    elif data_name == 'SMAP':
        n_fake = 8
    elif data_name == 'SMD':
        n_fake = 20
    elif data_name == 'SWAT':
        n_fake = 4
    else:
        n_fake = 4

    data_train, data_valid, data_test, n_features = get_loader_dataset(data_name, data_path, batch_size, win_size, step)
    label, score = train(n_features, embedding_size, num_hiddens, win_size, n_fake, dropout, device,
                         num_epochs, lr, weight_decay, patience, data_name, model_path,
                         data_train, data_valid, data_test, lambda1, lambda2)
    utils_.get_metrics(label, score, score_path, data_name)
