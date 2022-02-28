import torch
from torch import nn
from torch.nn.utils import spectral_norm
from extensions.emd import EMDLoss
from extensions.chamfer_dist import PatialChamferDistanceL1
from models.decoder import AttentionDecoder

from .build import MODELS


class PCN(nn.Module):
    def __init__(self, latent_size=1024):
        """
        mrc: boolean, True for enable missing region code
        """
        super().__init__()

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, latent_size, 1)
        )

    def forward(self, x):
        bs, n, _ = x.shape
        feature = self.first_conv(x.transpose(2, 1))  # B 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # B 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # B 512 n
        feature = self.second_conv(feature)  # B 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # B 1024

        return feature_global


class FeatureTransform(nn.Module):
    def __init__(self, latent_size=1024):
        """
        mrc: boolean, True for enable missing region code
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.BatchNorm1d(latent_size),
            nn.ReLU(inplace=True),
            nn.Linear(latent_size, latent_size),
            nn.BatchNorm1d(latent_size),
            nn.ReLU(inplace=True),
            nn.Linear(latent_size, latent_size)
        )
        self.correction = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.BatchNorm1d(latent_size),
            nn.ReLU(inplace=True),
            nn.Linear(latent_size, latent_size)
        )
    def transform_x(self, latent_x):
        latent_x = self.mlp(latent_x)
        latent_x = latent_x + self.correction(latent_x)
        return latent_x

    def transform_y(self, latent_y):
        latent_y = self.mlp(latent_y)
        return latent_y

    def forward(self, latent_x):
        return self.transform_x(latent_x)


class Discriminator(nn.Module):
    def __init__(self, latent_size=1024):
        """
        mrc: boolean, True for enable missing region code
        """
        super().__init__()
        self.mlp = nn.Sequential(
            spectral_norm(nn.Linear(latent_size, 1024)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(1024, 512)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, latent_x):
        return self.mlp(latent_x)


class FoldingDecoder(nn.Module):
    def __init__(self, latent_size=1024, num_output=2048):
        super().__init__()
        self.latent_size = latent_size
        self.num_output = num_output
        grid_size = 4 # set default
        self.grid_size = grid_size
        assert self.num_output% grid_size**2 == 0
        self.number_coarse = self.num_output // (grid_size ** 2 )

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_size,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,3*self.number_coarse)
        )
        self.final_conv = nn.Sequential(
            nn.Conv1d(1024+3+2,512,1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512,512,1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512,3,1)
        )
        a = torch.linspace(-0.05, 0.05, steps=grid_size, dtype=torch.float).view(1, grid_size).expand(grid_size, grid_size).reshape(1, -1).cuda()
        b = torch.linspace(-0.05, 0.05, steps=grid_size, dtype=torch.float).view(grid_size, 1).expand(grid_size, grid_size).reshape(1, -1).cuda()
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, grid_size ** 2) # 1 2 S
        # self.final_conv = nn.Sequential(
        #     nn.Linear(self.latent_size,1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024,1024,1),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024,3 * self.num_output)
        # )

    def forward(self, latent):
        bs = latent.size(0)
        # decoder
        coarse = self.mlp(latent).reshape(-1,self.number_coarse,3) # B M 3
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size**2, -1) # B M S 3
        point_feat = point_feat.reshape(-1,self.num_output, 3).transpose(2,1) # B 3 N

        seed = self.folding_seed.unsqueeze(2).expand(bs,-1,self.number_coarse, -1) # B 2 M S
        seed = seed.reshape(bs,-1,self.num_output)  # B 2 N

        latent = latent.unsqueeze(2).expand(-1,-1,self.num_output) # B 1024 N
        feat = torch.cat([latent, seed, point_feat], dim=1) # B C N

        fine = self.final_conv(feat) + point_feat   # B 3 N
        return fine.transpose(1,2).contiguous()

        # fine = self.final_conv(latent).reshape(bs, self.num_output, 3)
        # return fine



@MODELS.register_module()
class MFM(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.latent_dim = config.latent_dim
        self.num_pred = config.num_pred
        self.layers = config.layers

        self.build_modules(config)
        self.build_loss_func(config)

    def build_modules(self, config):
        self.feature_encoder = PCN(latent_size=self.latent_dim)

        self.mfm_modules = nn.ModuleList(
            [FeatureTransform(latent_size=self.latent_dim) for i in range(self.layers)]
        )
        # self.x_decoder = FoldingDecoder(latent_size=self.latent_dim, num_output=self.num_pred)
        # self.y_decoder = FoldingDecoder(latent_size=self.latent_dim, num_output=self.num_pred)
        # self.decoder = FoldingDecoder(latent_size=self.latent_dim, num_output=self.num_pred)
        self.decoder = AttentionDecoder(latent_dim=self.latent_dim, num_output=self.num_pred)
        self.discriminator_moudles = nn.ModuleList(
            [Discriminator(latent_size=self.latent_dim) for i in range(self.layers)]
        )

    def build_loss_func(self, config):
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()
        self.recon_criterion = EMDLoss()
        self.com_criterion = PatialChamferDistanceL1()

    def gen_params(self):
        params = list(self.feature_encoder.parameters())
        params += list(self.mfm_modules.parameters())
        # params += list(self.x_decoder.parameters())
        # params += list(self.y_decoder.parameters())
        params += list(self.decoder.parameters())
        return params

    def dis_params(self):
        params = list(self.discriminator_moudles.parameters())
        return params

    def dis_forward(self, x, y):
        latent_x = [self.feature_encoder(x)]
        latent_y = [self.feature_encoder(y)]
        for i in range(self.layers):
            latent_x.append(self.mfm_modules[i].transform_x(latent_x[i]))
            latent_y.append(self.mfm_modules[i].transform_y(latent_y[i]))

        x_logits = []
        y_logits = []
        for i in range(self.layers):
            x_logits.append(self.discriminator_moudles[i](latent_x[i+1]))
            y_logits.append(self.discriminator_moudles[i](latent_y[i+1]))

        return x_logits, y_logits

    def gen_forward(self, x, y):
        latent_x = self.feature_encoder(x)
        latent_y = self.feature_encoder(y)
        x_logits = []
        for i in range(self.layers):
            latent_x = self.mfm_modules[i].transform_x(latent_x)
            latent_y = self.mfm_modules[i].transform_y(latent_y)
            x_logits.append(self.discriminator_moudles[i](latent_x))
        # x2y = self.x_decoder(latent_x)
        # y2y = self.y_decoder(latent_y)
        x2y = self.decoder(latent_x)
        y2y = self.decoder(latent_y)
        return x2y, y2y, x_logits

    def get_dis_loss(self, ret):
        x_logits, y_logits =  ret
        bs = x_logits[0].size(0)
        k = len(x_logits)
        loss_disc = 0
        for i in range(k):
            loss_disc += self.bce_criterion(y_logits[i], torch.ones(bs, 1).to(y_logits[i])) + self.bce_criterion(x_logits[i], torch.zeros(bs, 1).to(x_logits[i]))
        loss_disc = loss_disc / (2 * k)
        return loss_disc

    def get_gen_loss(self, ret, gt):
        x2y, y2y, x_logits = ret
        x, y = gt
        # Reconstruction Loss
        recon_loss = self.recon_criterion(y2y, y).mean() * 100
        # Completion Loss
        com_loss = self.com_criterion(x, x2y).mean() * 100
        # Feature Matching Loss
        bs = x_logits[0].size(0)
        k = len(x_logits)
        matching_loss = 0
        for i in range(k):
            matching_loss += self.bce_criterion(x_logits[i], torch.zeros(bs, 1).to(x_logits[i]))
        matching_loss = matching_loss / k
        loss_gen = recon_loss + com_loss + matching_loss
        return loss_gen, recon_loss, com_loss, matching_loss

    def forward(self, x):
        latent_x = self.feature_encoder(x)
        for i in range(self.layers):
            latent_x = self.mfm_modules[i].transform_x(latent_x)
        # x2y = self.x_decoder(latent_x)
        x2y = self.decoder(latent_x)
        return x2y