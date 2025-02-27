#Essentials
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from umap import UMAP

# Defining Autoencoder with batch information class
class BatchAutoencoder(nn.Module):
    def __init__(self, 
                 d_z = 10, 
                 input_size = 1024, 
                 batch_size = 10):
        super().__init__()
        self.d_z = d_z
        self.input_size = input_size
        self.batch_size = batch_size
        self.encoder = nn.Sequential(
            nn.Linear(input_size + batch_size, 128),
            nn.Linear(128, 64),
            nn.Linear(64, d_z)
        )

        self.decoder = nn.Sequential(
            nn.Linear(d_z, 64),
            nn.Linear(64, 128),
            nn.Linear(128, input_size)
        )

    def forward(self, x):
        x = x.view(-1, self.input_size + self.batch_size)
        z = self.encoder(x)
        decoded = self.decoder(z)
        return decoded.view(-1, self.input_size)
    
    def encode(self, x):
        x = x.view(-1, self.input_size + self.batch_size)
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def loss(self, out, x):
        x = x.view(-1, self.input_size + self.batch_size)
        x = x[:, :-self.batch_size]
        return nn.MSELoss()(out, x)

# Defining Batch Discriminator class
class BatchDiscriminator(nn.Module):
    def __init__(self,
                 input_size = 1024,
                 hl1_size = 128,
                 hl2_size = 64,
                 batch_size = 10,
                 tissue_size = 2):
        super().__init__()
        self.input_size = input_size
        self.hl1_size = hl1_size
        self.hl2_size = hl2_size
        self.batch_size = batch_size
        self.tissue_size = tissue_size
        self.ffnn = nn.Sequential(
            nn.Linear(input_size + tissue_size, hl1_size),
            nn.ReLU(),
            nn.Linear(hl1_size, hl2_size),
            nn.ReLU(),
            nn.Linear(hl2_size, batch_size)
        )

    def forward(self, x):
        x = x.view(-1, self.input_size + self.tissue_size)
        y = self.ffnn(x)
        return y
    
    def loss(self, out, y):
        return nn.CrossEntropyLoss()(out, y)

# Defining tissue classifier class
class TissueClassifier(nn.Module):
    def __init__(self,
                 input_size,
                 hl1_size = 128,
                 hl2_size = 64,
                 tissue_size = 2):
        super().__init__()
        self.input_size = input_size
        self.hl1_size = hl1_size
        self.hl2_size = hl2_size
        self.tissue_size = tissue_size
        self.ffnn = nn.Sequential(
            nn.Linear(input_size, hl1_size),
            nn.ReLU(),
            nn.Linear(hl1_size, hl2_size),
            nn.ReLU(),
            nn.Linear(hl2_size, tissue_size)
        )

    def forward(self, x):
        x = x.view(-1, self.input_size)
        y = self.ffnn(x)
        return y
    
    def loss(self, out, y):
        return nn.CrossEntropyLoss()(out, y)

# Define discriminator that differentiates between original data from recreated data
# This assures that the original data distribution is conserved
class DataDiscriminator(nn.Module):
    def __init__(self,
                 input_size,
                 exp_size,
                 hl1_size = 128,
                 hl2_size = 64,
                 output_size = 2):
        super().__init__()
        self.input_size = input_size + exp_size
        self.hl1_size = hl1_size
        self.hl2_size = hl2_size
        self.output_size = output_size
        self.ffnn = nn.Sequential(
            nn.Linear(input_size + exp_size, hl1_size),
            nn.ReLU(),
            nn.Linear(hl1_size, hl2_size),
            nn.ReLU(),
            nn.Linear(hl2_size, output_size)
        )

    def forward(self, x):
        x = x.view(-1, self.input_size)
        y = self.ffnn(x)
        return y
    
    def loss(self, out, y):
        return nn.CrossEntropyLoss()(out, y)


# Defining ABaCo algorithm
class ABaCo(nn.Module):

    #ABaCo core part is a regular autoencoder
    def __init__(self, 
                 d_z = 10, 
                 input_size = 1024, 
                 batch_size = 10):
        super().__init__()
        self.d_z = d_z
        self.input_size = input_size
        self.batch_size = batch_size
        self.encoder = nn.Sequential(
            nn.Linear(input_size + batch_size, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, d_z)
        )

        self.decoder = nn.Sequential(
            nn.Linear(d_z, 128),
            nn.Linear(128, 256),
            nn.Linear(256, 512),
            nn.Linear(512, input_size)
        )

    def forward(self, x):
        x = x.view(-1, self.input_size + self.batch_size)
        z = self.encoder(x)
        decoded = self.decoder(z)
        return decoded.view(-1, self.input_size)
    
    def encode(self, x):
        x = x.view(-1, self.input_size + self.batch_size)
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    # Define the classifier and discriminator models
    def train_model(self,
                    batch_model: BatchDiscriminator,
                    out_class_model: TissueClassifier,
                    latent_class_model: TissueClassifier,
                    data_disc_model: DataDiscriminator,
                    train_loader,
                    ohe_exp_loader,
                    num_epochs,
                    device,
                    w_recon = 1.0,
                    w_adver = 1.0,
                    w_latent = 1.0,
                    w_output = 1.0,
                    w_disc = 1.0,
                    w_data = 1.0,
                    lr_recon = 1e-5,
                    lr_adver = 1e-5,
                    lr_latent = 1e-5,
                    lr_output = 1e-5,
                    lr_disc = 1e-5,
                    lr_data = 1e-5,
                    val_loader = None,
                    test_loader = None,
                    model_name = "model",
                    save_model = False
                    ):
        """
        Placeholder
        """
        train_dis_losses = []
        train_adv_losses = []
        train_recon_losses = []
        train_latent_losses = []
        train_output_losses = []
        train_data_losses = []
        test_losses = []
        val_losses = []
        lowest_val_loss = float('inf')
        best_epoch = -1
        best_model_state = None

        #Optimizer for discriminator only
        step_1_optimizer = torch.optim.Adam(batch_model.parameters(), lr = lr_disc, weight_decay=1e-5)
        #Optimizer for adversarial training (autoencoder only)
        step_2_optimizer = torch.optim.Adam(self.parameters(), lr = lr_adver, weight_decay=1e-5)
        #Optimizer for biological conservation (autoencoder and classifiers)
        step_3_1_optimizer = torch.optim.Adam(self.parameters(), lr = lr_recon, weight_decay=1e-5)
        step_3_2_optimizer = torch.optim.Adam(latent_class_model.parameters(), lr = lr_latent, weight_decay=1e-5)
        step_3_3_optimizer = torch.optim.Adam(out_class_model.parameters(), lr = lr_output, weight_decay=1e-5)
        #v2 - Optimizer for discriminator of original distribution data and adversarial training AE
        step_4_optimizer = torch.optim.Adam(data_disc_model.parameters(), lr = lr_data, weight_decay=1e-5)
        step_5_optimizer = torch.optim.Adam(self.parameters(), lr = lr_adver, weight_decay=1e-5)

        #Loss function for discriminator only
        step_1_criterion = nn.CrossEntropyLoss()
        #Adversarial loss function of autoencoder
        step_2_criterion = nn.KLDivLoss()
        #Biological conservation loss functions
        step_3_1_criterion = nn.MSELoss()
        step_3_2_criterion = nn.CrossEntropyLoss()
        step_3_3_criterion = nn.CrossEntropyLoss()
        #Original data distribution conservation loss functions
        step_4_criterion = nn.CrossEntropyLoss()
        step_5_criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            #Training
            self.train()
            batch_model.train()
            out_class_model.train()
            latent_class_model.train()
            train_dis_loss = 0.0
            train_adv_loss = 0.0
            train_recon_loss = 0.0
            train_output_loss = 0.0
            train_latent_loss = 0.0
            train_data_loss = 0.0

            for (x, y, k), (ohe_exp) in zip(train_loader, ohe_exp_loader):
                #Forward pass to Autoencoder
                x = x.to(device)
                z = self.encode(x)
                out_ae = self.decode(z)

                #Forward pass to discriminator
                input_disc = torch.concat((out_ae, ohe_exp), 1) # reconstructed otus + tissue ohe
                out_batch_class = batch_model(input_disc)

                #1st Backpropagation: Discriminator
                step_1_loss = w_disc * step_1_criterion(out_batch_class, x[:, -self.batch_size:])
                step_1_optimizer.zero_grad()
                step_1_loss.backward(retain_graph=True)
                step_1_optimizer.step()
                out_batch_class = out_batch_class.detach()
                out_batch_class = batch_model(input_disc)
                out_ae = out_ae.detach()
                z = z.detach()
                z = self.encode(x)
                out_ae = self.decode(z)

                #2nd Backpropagation: Adversarial AE
                target_dist = torch.full_like(out_batch_class, 1/self.batch_size)
                out_batch_prob = torch.log_softmax(out_batch_class, dim=1)
                step_2_loss = w_adver * step_2_criterion(out_batch_prob, target_dist)
                step_2_optimizer.zero_grad()
                step_2_loss.backward(retain_graph=True)
                step_2_optimizer.step()
                out_batch_class = out_batch_class.detach()
                out_batch_class = batch_model(input_disc)
                out_ae = out_ae.detach()
                z = z.detach()
                z = self.encode(x)
                out_ae = self.decode(z)

                #Forward pass to classificators
                out_out_class = out_class_model(out_ae)     #this classifies using AE output
                out_latent_class = latent_class_model(z)    #this classifies using AE latent space

                #3rd Backpropagation: Triple loss function
                # step_3_1_loss = step_3_1_criterion(out_ae, x[:,:-self.batch_size])
                step_3_1_loss = 0
                for i in range(x.shape[0]):
                    k_i = k[i].item()
                    step_3_1_loss += step_3_1_criterion(out_ae[i, :k_i], x[i, :k_i]) # Only compute loss on features values (not zero padding values)

                step_3_1_loss /= x.shape[0] # Normalize for the batch size
                step_3_2_loss = step_3_2_criterion(out_latent_class, y)
                step_3_3_loss = step_3_3_criterion(out_out_class, y)
                step_3_loss = w_recon * step_3_1_loss + w_latent * step_3_2_loss + w_output * step_3_3_loss
                step_3_1_optimizer.zero_grad()
                step_3_2_optimizer.zero_grad()
                step_3_3_optimizer.zero_grad()
                step_3_loss.backward()
                step_3_1_optimizer.step()
                step_3_2_optimizer.step()
                step_3_3_optimizer.step()
                out_ae = out_ae.detach()
                z = z.detach()
                z = self.encode(x)
                out_ae = self.decode(z)

                #4th Backpropagation: Training data discriminator
                original_data = torch.concat((x[:, :-self.batch_size], ohe_exp), 1)
                recon_data = torch.concat((out_ae, ohe_exp), 1)
                out_real_data_class = data_disc_model(original_data)
                out_recon_data_class = data_disc_model(recon_data)

                #Original data class
                original_class = torch.zeros(x.shape[0], 2, device=device)
                original_class[:x.shape[0]] = torch.tensor([1, 0], device=device)

                #Reconstructed data class
                recon_class = torch.zeros(x.shape[0], 2, device=device)
                recon_class[:x.shape[0]] = torch.tensor([0, 1], device=device)
                
                #Compute loss for data discriminator
                step_4_loss = w_data * step_4_criterion(out_real_data_class, original_class)
                step_4_loss += w_data * step_4_criterion(out_recon_data_class, recon_class)

                #Backpropagation
                step_4_optimizer.zero_grad()
                step_4_loss.backward()
                step_4_optimizer.step()

                #Detach everything for next pass
                out_ae = out_ae.detach()
                z = z.detach()
                z = self.encode(x)
                out_ae = self.decode(z)

                #Recompute for following backpropagation
                recon_data = torch.concat((out_ae, ohe_exp), 1)
                out_recon_data_class = data_disc_model(recon_data)

                #5th Backpropagation: Adversarial training of AE from data discriminator (only computed with reconstructed data output)
                step_5_loss = w_adver * step_5_criterion(out_recon_data_class, original_class)
                step_5_optimizer.zero_grad()
                step_5_loss.backward()
                step_5_optimizer.step()
                
                #Save loss values
                train_dis_loss += step_1_loss.item()
                train_adv_loss += step_2_loss.item()
                train_recon_loss += step_3_1_loss.item()
                train_latent_loss += step_3_2_loss.item()
                train_output_loss += step_3_3_loss.item()
                train_data_loss += step_4_loss.item()

            train_dis_loss /= len(train_loader)
            train_adv_loss /= len(train_loader)
            train_recon_loss /= len(train_loader)
            train_latent_loss /= len(train_loader)
            train_output_loss /= len(train_loader)
            train_data_loss /= len(train_loader)
            
            train_dis_losses.append(train_dis_loss)
            train_adv_losses.append(train_adv_loss)
            train_recon_losses.append(train_recon_loss)
            train_latent_losses.append(train_latent_loss)
            train_output_losses.append(train_output_loss)
            train_data_losses.append(train_data_loss)
            
            if (epoch + 1) % 1 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} | Dis. Train Loss: {train_dis_loss:.4f} | Adv. Train Loss: {train_adv_loss:.4f} | Recon. Train Loss: {train_recon_loss:.4f} | Lat. Train Loss: {train_latent_loss:.4f} | Out. Train Loss: {train_output_loss:.4f} | Data Train Loss: {train_data_loss:.4f}")

        return train_dis_losses, train_adv_losses, train_recon_losses