import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss
import torch.optim as optim
import torch.utils.data as utils
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from VAEGAN import VAE
from net import Discriminator
from copy import deepcopy
import matplotlib.pyplot as plt
import random
import time
import cv2

device = "cuda:0" if torch.cuda.is_available() else "cpu"
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)


class VAECycleGan(nn.Module):
    
    def __init__(self, args):
        super(VAECycleGan, self).__init__()
        self.x_dim = args["x_dim"]
        self.z_dim = args["z_dim"]
        self.lam0 = args["lam0"]
        self.lam1 = args["lam1"]
        self.lam2 = args["lam2"]
        self.lam3 = args["lam3"]
        self.lam4 = args["lam4"]
        
        self.vae1 = VAE(self.x_dim, h_dim1=512, h_dim2=256, h_dim3=128, z_dim=self.z_dim).to(device)
        self.vae2 = VAE(self.x_dim, h_dim1=512, h_dim2=256, h_dim3=128, z_dim=self.z_dim).to(device)
        # self.share_vae_features()
        
        self.young_data_fname = "../data1.csv"
        self.old_data_fname = "/data2.csv"
        self.young_data = np.genfromtxt(self.young_data_fname, delimiter=",").transpose()[1:, 1:]
        self.old_data = np.genfromtxt(self.old_data_fname, delimiter=",").transpose()[1:, 1:]
        
        self.young_data = torch.from_numpy(self.young_data).to(device).float()
        self.old_data = torch.from_numpy(self.old_data).to(device).float()
        
        self.young_train, self.young_test = self.split_data(self.young_data, 0.1)
        self.old_train, self.old_test = self.split_data(self.old_data, 0.1)
        
        self.young_ds = utils.TensorDataset(self.young_train)
        self.young_test_ds = utils.TensorDataset(self.young_test)
        self.young_dataloader = utils.DataLoader(self.young_ds, batch_size=10, shuffle=True, num_workers= 0)
        self.young_test_loader = utils.DataLoader(self.young_test_ds, batch_size=10, shuffle=True, num_workers= 0)
        self.old_ds = utils.TensorDataset(self.old_train)
        self.old_test_ds = utils.TensorDataset(self.old_test)
        self.old_dataloader = utils.DataLoader(self.old_ds, batch_size=10, shuffle=True, num_workers= 0)
        self.old_test_loader = utils.DataLoader(self.old_test_ds, batch_size=10, shuffle=True, num_workers= 0)
        
        
        self.VAE_optim = optim.Adam(list(self.vae1.parameters()) + list(self.vae2.parameters()), lr=0.005)
    # self.VAE_optim = optim.Adam(list(self.vae1.parameters()) + list(self.vae2.parameters()))
    # self.VAE_optim = optim.Adam(self.vae1.parameters(), lr=0.001)
    
    def split_data(self, data, p_test):
        
        N = len(data)
        inds = list(range(N))
        random.shuffle(inds)
        
        train_N = int((1 - p_test) * N)
        test_N = int((p_test * N))
        
        train = data[:train_N]
        test = data[train_N:]
        
        return train, test
    
    # def share_vae_features(self):
    # self.vae1.fc31 = self.vae2.fc31
    # self.vae1.fc32 = self.vae2.fc32
    # self.vae1.fc4 = self.vae2.fc4
    
    def VAELoss(self, x_in, y_in):
        """
            BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            return BCE + KLD
            """
        G1_out, mu1, log_var1 = self.vae1(x_in)
        KLD = -0.5 * (torch.sum(1 + log_var1 - mu1.pow(2) - log_var1.exp()))
        #KLD = -self.lam1 *( torch.sum(1 + log_var1 - mu1.pow(2) - log_var1.exp()))
        BCE = self.lam2 * (F.mse_loss(G1_out, x_in.view(-1, self.x_dim), reduction='sum'))
        #BCE =  F.mse_loss(G1_out, x_in.view(-1, self.x_dim), reduction='sum')
        L1 = (BCE + self.lam1 * KLD) / x_in.shape[0]
        # print(G1_out.shape, log_var1.shape, mu1.pow(2).shape)
        
        G2_out, mu2, log_var2 = self.vae2(y_in)
        KLD_2 = -0.5 * (torch.sum(1 + log_var2 - mu2.pow(2) - log_var2.exp()))
        #KLD_2 = -self.lam1 *( torch.sum(1 + log_var2 - mu2.pow(2) - log_var2.exp()))
        BCE_2 = self.lam2 * (F.mse_loss(G2_out, y_in.view(-1, self.x_dim), reduction='sum'))
        #BCE_2 =  F.mse_loss(G2_out, y_in.view(-1, self.x_dim), reduction='sum')
        L = L1 + (BCE_2 + self.lam1 * KLD_2) / y_in.shape[0]
        L.backward()
        
        return L
    
    def train(self, num_epochs):
        
        self.vae1.train()
        self.vae2.train()
        
        losses = []
        
        for i in range(num_epochs):
            epoch_loss = 0.0
            total_vae = 0.0
            # total_D = 0.0
            # total_G = 0.0
            # total_cc = 0.0
            
            if (i == 30):
                self.VAE_optim = optim.Adam(list(self.vae1.parameters()) + list(self.vae2.parameters()), lr=0.001)
        
            train_steps = min(len(self.old_dataloader), len(self.young_dataloader))
            
            self.young_dataloader = utils.DataLoader(self.young_ds, batch_size=10, shuffle=True, num_workers= 0)
            self.old_dataloader = utils.DataLoader(self.old_ds, batch_size=10, shuffle=True, num_workers= 0)
            old_data = iter(self.old_dataloader)
            young_data = iter(self.young_dataloader)
            
            # Iterate through each batch
            for j in range(train_steps - 1):
                # for j in range(10):
                # Get batch of data
                [young_cells] = next(young_data)
                [old_cells] = next(old_data)
                
                young_cells = young_cells.to(device).flatten(1, -1)
                old_cells = old_cells.to(device).flatten(1, -1)
                # Zero out all optimizers
                self.VAE_optim.zero_grad()
                
                # vaeloss = torch.Tensor([0])
                
                self.VAE_optim.zero_grad()
                vaeloss = self.VAELoss(young_cells, old_cells)
                self.VAE_optim.step()
                
                # Exclude discriminator loss from total
                loss = vaeloss
                
                epoch_loss += loss.item()
                total_vae += vaeloss.item()
            
            [epoch_loss, total_vae] = loss_arr = np.array([epoch_loss, total_vae]) / train_steps
            
            print("Losses at epoch %d\t VAE: %f\tTOTAL:  %f" % (i + 1, total_vae, epoch_loss))
            
            losses.append(list(loss_arr))
            plt.plot(np.array(losses)[:, 0])
            plt.show(block=False)
            plt.pause(0.001)
            
                    self.test(save=False)

fig = plt.figure()
losses = np.array(losses)
# [v, _, _, c] = plt.plot(losses[:,1:])
[vae_loss] = plt.plot(losses[:, [1]])
    plt.legend([vae_loss], ["VAE Reconstruction"], loc=1)
    
    plt.show()
        fig.savefig("/content/drive/My Drive/NewVAE/TechvarResults/64_D/2.0N1000_KL_Loss_20000.png")

    def test(self, save=True):
        
        self.vae1.eval()
        self.vae2.eval()
        
        young_mu, young_logvar = self.vae1.encode(self.young_data)
        young_Z = self.vae1.sampling(young_mu, young_logvar)
        young_output = self.vae1.decode(young_Z)
        young_corr = self.pearson_correlation(self.young_data, young_output)
        
        old_mu, old_logvar = self.vae2.encode(self.old_data)
        old_Z = self.vae2.sampling(old_mu, old_logvar)
        old_output = self.vae2.decode(old_Z)
        old_corr = self.pearson_correlation(self.old_data, old_output)
        
        print("old corr: ", old_corr, " young corr: ", young_corr)
        
        if (save):
            # np.savetxt(
            #  "/content/drive/My Drive/NewVAE/TechvarResults/2_D/observed_counts_logNorm_5pop_continuous_A_depth-10000_alpha-0.005_mu.csv",
            #  young_mu.cpu().data.numpy(), delimiter='\t')
            #np.savetxt(
            #   "/content/drive/My Drive/NewVAE/TechvarResults/2_D/observed_counts_logNorm_5pop_continuous_A_depth-10000_alpha-0.005_logvar.csv",
            #  young_logvar.cpu().data.numpy(), delimiter='\t')
            np.savetxt(
                       "/latent1.csv",
                       young_Z.cpu().data.numpy(), fmt="%2.7g", delimiter=',')
                       #np.savetxt(
                       #   "/content/drive/My Drive/NewVAE/TechvarResults/2_D/observed_counts_logNorm_5pop_continuous_A_depth-10000_alpha-0.005_correlation.csv",
                       #   np.array([young_corr.cpu().data.numpy()]), delimiter='\t')
                       np.savetxt(
                                  "data1recreated_from_vae.csv",
                                  young_output.cpu().data.numpy(), fmt="%2.7g", delimiter=',')
                       
                       #np.savetxt("/content/drive/My Drive/NewVAE/TechvarResults/2_D/observed_counts_logNorm_5pop_continuous_A_depth-10000_alpha-0.1_mu.csv",
                       #old_mu.cpu().data.numpy(), delimiter='\t')
                       #np.savetxt("/content/drive/My Drive/NewVAE/TechvarResults/2_D/observed_counts_logNorm_5pop_continuous_A_depth-10000_alpha-0.1_logvar.csv",
                       #old_logvar.cpu().data.numpy(), delimiter='\t')
                       np.savetxt("latent2.csv",
                                  old_Z.cpu().data.numpy(), fmt="%2.7g",delimiter=',')
                       #np.savetxt("/content/drive/My Drive/NewVAE/TechvarResults/2_D/observed_counts_logNorm_5pop_continuous_A_depth-10000_alpha-0.1_correlation.csv",
                       #np.array([old_corr.cpu().data.numpy()]), delimiter='\t')
                       np.savetxt("data2recreated_from_vae.csv",
                                  old_output.cpu().data.numpy(),fmt="%2.7g", delimiter=',')



def pearson_correlation(self, x, y):
    normx = x - torch.mean(x)
    normy = y - torch.mean(y)
    
    return torch.mean(torch.sum(normx * normy, dim=1) / (
                                                         torch.sqrt(torch.sum(normx ** 2, dim=1)) * torch.sqrt(torch.sum(normy ** 2, dim=1))))


def main():
    args = {
        "x_dim": 3000,
        "z_dim": 64,
        "lam0": 5.0,  # Gan loss
        "lam1": 1.0,  # VAE KL loss #0.01
        "lam2": 1.0,  # VAE match loss
        "lam3": 0.01,  # CYCLEGAN KL loss
        "lam4": 5.0  # CYCLEGAN match loss
    }
    net = VAECycleGan(args)
    net.train(num_epochs=1200)
    net.test()

#torch.save(net.state_dict(), "/content/drive/My Drive/NewVAE/TechvarResults/64_D/0.5N8_KL_model_20000.pth")


if __name__ == "__main__":
    main()



