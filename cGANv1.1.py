#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 15:04:31 2023

@author: ruby
"""

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
import torch
from torch import nn, optim
from torchvision import transforms
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import time
from astropy.io import fits
from astropy.visualization import make_lupton_rgb

# path to working directory
home = "/Users/ruby/Documents/Python Scripts/Filters/"

# total number of cutouts
f115w_path = home+'F115W/'
f150w_path = home+'F150W/'
f200w_path = home+'F200W/'
f277w_path = home+'F277W/'
f356w_path = home+'F356W/'
f444w_path = home+'F444W/'
total_inputs = len(os.listdir(f115w_path)+os.listdir(f150w_path)+os.listdir(f200w_path)
                   +os.listdir(f277w_path)+os.listdir(f356w_path)+os.listdir(f444w_path))
    

# list of filters which is then split into blue filters and red filters
filters = ['F115W/', 'F150W/', 'F200W/', 'F277W/', 'F356W/', 'F444W/']
nbands = len(filters)

SIZE = 256    
# normalise the values of each fits file between 0 and 1
def Normalise(data, lower=0, upper=1):
    return ((data - data.min())/ (data.max() - data.min()))


# create the dataset for the fits files
class FilterDataset(Dataset):
    def __init__(self, path):
        ''' path = path to directory containing fits files '''
        self.path = path
        self.transforms_inputs = torch.nn.Sequential(transforms.Resize((SIZE, SIZE)),
                                                     transforms.Grayscale(num_output_channels=1))
        self.transforms_labels = torch.nn.Sequential(transforms.Resize((SIZE, SIZE)),
                                                     transforms.Grayscale(num_output_channels=1))
        self.f115w_path = path+'F115W/'
        self.f150w_path = path+'F150W/'
        self.f200w_path = path+'F200W/'
        self.f277w_path = path+'F277W/'
        self.f356w_path = path+'F356W/'
        self.f444w_path = path+'F444W/'
        self.l1 = len(os.listdir(self.f115w_path))
    
    def __len__(self):
        # return total number of fits files for the galaxy cutouts consistent
        # with the 'idx' in the __getitem__ method
        return (self.l1)
    
    def __getitem__(self, idx):
        # get the name of the fits file
        name = str(idx)+'galaxy_cutout.fits'
        # for each input filter, get each fits file, open and extract the 
        # first row (only row which is 'SCI' data) and normalise before
        # formatting into an array
        hdu1 = fits.open(self.f115w_path+name)[0]
        data1 = Normalise(hdu1.data)
        data1 = np.array(data1)
        hdu2 = fits.open(self.f150w_path+name)[0]
        data2 = Normalise(hdu2.data)
        data2 = np.array(data2)
        hdu3 = fits.open(self.f200w_path+name)[0]
        data3 = Normalise(hdu3.data)
        data3 = np.array(data3)

        # now the same for the label filters as 
        hdu4 = fits.open(self.f277w_path+name)[0]
        data4 = Normalise(hdu4.data)
        data4 = np.array(data4)
        hdu5 = fits.open(self.f356w_path+name)[0]
        data5 = Normalise(hdu5.data)
        data5 = np.array(data5)
        hdu6 = fits.open(self.f444w_path+name)[0]
        data6 = Normalise(hdu6.data)
        data6 = np.array(data6)
        
        # stack the input filters (f115w, f150w, f200w)
        inputs = np.stack((data1, data2, data3)).astype("float32")
        # reformat the inputs as tensors
        inputs = transforms.ToTensor()(inputs)
        # reshape the tensor to [C, H, W] for the transform to work
        inputs = inputs.permute(1,0,2)
        # transform the input tensor by resizing to (256,256) and changing
        # the number of output channels to 1
        inputs = self.transforms_inputs(inputs)
        # do the same for the labels
        labels = np.stack((data4, data5, data6)).astype("float32")
        labels = transforms.ToTensor()(labels)
        labels = labels.permute(1,0,2)
        labels = self.transforms_labels(labels)
        # since the labels need to have 2 C channels for the network,
        # repeat the grayscale channel twice 
        labels = labels.repeat(2,1,1)
        
        # return the inputs with corresponding labels in a dictionary
        return {'Inputs': inputs, 'Labels': labels}

# split the generated dataset into training and testing with a validation 
# split of 10%
dataset = FilterDataset(path=home)                  # generate labelled dataset
BATCH_SIZE = 16                                     # set the batch size
VALIDATION_SPLIT = 0.1                              # set the validation split of 10%
SHUFFLE_DATASET = True                              # shuffle the training data only
RANDOM_SEED = 42                                    # randomly shuffle through indexed dataset

# create indices for training and test split
DATASET_SIZE = len(dataset)
# list the dataset with an index for each entry
indices = list(range(DATASET_SIZE))
# define the split for the dataset
split = int(np.floor(DATASET_SIZE * VALIDATION_SPLIT))
if SHUFFLE_DATASET:
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices)
# split the dataset into training and testing 
train_indices, test_indices = indices[split:], indices[:split]

# create data samplers and dataloaders
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
# create dataloaders
trainloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
testloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_sampler)
# print(len(trainloader), len(testloader)) = 110, 13

data = next(iter(trainloader))
inputs_, labels_ = data['Inputs'], data['Labels']
# print(inputs_.shape, labels_.shape) = torch.Size([16, 1, 256, 256]) torch.Size([16, 2, 256, 256])

# Generator as proposed by the pix2pix image translation paper
class UnetBlock(nn.Module):
    ''' U-Net is used as the generator of the GAN.
        Creates the U-Net from the middle part down and adds down-sampling and
        up-sampling modules to the left and right of the middle module.
        8 layers down so start with a 256x256 tensor with 1 channel, down-sample 
        to a 1x1 tensor, then up-sample to a 256x256 tensor with 2 channels. '''
    def __init__(self, nf, ni, submodule=None, input_c=None, dropout=False,
                 innermost=False, outermost=False):
        ''' ni = number of filters in the inner convolution layer
            nf = number of filters in the outer convolution layer
            input_c = number of input channels (= 1)
            submodule = previously defined submodules
            dropout = not using dropout layers '''
        super().__init__()
        self.outermost = outermost
        if input_c is None: input_c = nf
        downconv = nn.Conv2d(in_channels=input_c, out_channels=ni, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(ni)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(nf)
        
        if outermost: # if this module is the outermost module
            upconv = nn.ConvTranspose2d(in_channels=ni*2, out_channels=nf, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost: # if this module is the innermost module
            upconv = nn.ConvTranspose2d(in_channels=ni, out_channels=nf, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(in_channels=ni*2, out_channels=nf, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if dropout: up += [nn.Dropout(0.5)]
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else: # add skip connections
            return torch.cat([x, self.model(x)], dim=1)

class Unet(nn.Module):
    ''' U-Net based generator.'''
    def __init__(self, input_c=1, output_c=2, n_down=8, num_filters=64):
        ''' input_c = number of input channels (= 1)
            output_c = number of output channels (= 2)
            n_down = number of downsamples: we start with 256x256 and after 
                                            8 layers, we have a 1x1 tensor at the bottleneck.
            num_filters = number of filters in the last convolution layer. '''
        super().__init__()
        unet_block = UnetBlock(num_filters*8, num_filters*8, innermost=True)
        for _ in range(n_down - 5):
            # adds intermediate layers with num_filters * 8 filters
            unet_block = UnetBlock(num_filters*8, num_filters*8, submodule=unet_block, dropout=True)
        out_filters = num_filters*8
        for _ in range(3):
            # gradually reduce the number of filters to num_filters
            unet_block = UnetBlock(out_filters//2, out_filters, submodule=unet_block)
            out_filters //= 2
        self.model = UnetBlock(output_c, out_filters, input_c=input_c, submodule=unet_block, outermost=True)
    
    def forward(self, x):
        return self.model(x)
            
# Now the discriminator as proposed by the paper 
class PatchDiscriminator(nn.Module):
    ''' Patch discriminator stacks blocks of convolution-batchnorm-leakyrelu 
        to decide whether the input tensor is real or fake. 
        Patch discriminator outputs one number for every NxN pixels of the input
        and decides whether each "patch" is real/fake. 
        Patches will be 70 by 70. '''
    def __init__(self, input_c, num_filters=64, n_down=3):
        ''' input_c = number of input channels (= 1)
            num_filters = number of filters in last convolution layer
            n_down = number of layers '''
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        # use if statement to take care of not using a stride of 2 in the last block of the loop
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i+1), s=1 if i == (n_down-1) else 2) for i in range(n_down)]
        # do not use normalisation or activation for the last layer of the model
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False, act=False)] # ouput 1 channel prediction
        self.model = nn.Sequential(*model)
    
    # make a separate method for the repetitive layers
    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True):
        ''' norm = batch norm layer
        act = apply activation '''
        layers = [nn.Conv2d(in_channels=ni, out_channels=nf, kernel_size=k, stride=s, padding=p, bias=not norm)]
        if norm: layers += [nn.BatchNorm2d(nf)]
        if act: layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)            
            
# discriminator = PatchDiscriminator(2)
# dummy_variable = torch.randn(16, 2, 256, 256)
# out = discriminator(dummy_variable)  
# out.shape = torch.Size([16, 1, 256, 256])       
        
# Unique loss function for the GAN 
class GANLoss(nn.Module):
    ''' Calculates the GAN loss of the final model.
        Uses a "vanilla" loss and registers constant tensors for the real
        and fake labels. Returns tensors full of zeros or ones to compute the loss'''
        
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer(name='real_label', tensor=torch.tensor(real_label))
        self.register_buffer(name='fake_label', tensor=torch.tensor(fake_label))
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss() # binary cross entropy loss
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss() # don't use this
        
    def get_labels(self, preds, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds) # expand to the same size as predictions
    
    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss
    
# initilise the model here
def Init_Weights(net, init='norm', gain=.02):
    ''' Image-to-image translation paper state that the model is initialised 
        with a mean of 0.0 and std 0.02'''
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                # fills tensor with values drawn from normal distribution N(mean,std^2)
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier': # taken from a paper
            # fills input tensor with avlues sampled from N(0,std^2)
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming': # taken from a paper
                # resulting tensor has values sampled from N(0,std^2)
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(tensor=m.bias.data, val=0.0) # tensor filled with zeros
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(tensor=m.bias.data, val=0.0)
    
    net.apply(init_func)
    print(f"model initialised with {init} initialisation")
    return net

def Init_Model(model, device):
    model = model.to(device)
    model = Init_Weights(model)
    return model

  
# now to initialise the main GAN network
class GANModel(nn.Module):
    ''' Initialises the model defining the generator and discriminator in the
        __init__ function using the functions given and initialises the loss
        functions '''
    def __init__(self, net_G=None, lr_G=2e-4, lr_D=2e-4, beta1=.5, beta2=.999, lambda_L1=100.): 
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1
        
        if net_G is None:
            self.net_G = Init_Model(Unet(input_c=1, output_c=2, n_down=8, num_filters=64), self.device)
        else:
            self.net_G.to(self.device)
        
        self.net_D = Init_Model(PatchDiscriminator(input_c=3, n_down=3, num_filters=64), self.device)
        self.GAN_loss = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1_loss = nn.L1Loss()
        # initialise optimisers for generator and discriminator using Adam optimiser
        # and parameters stated in the paper 
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1,beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1,beta2))
        # initialise empty lists to append the generator and discriminator losses to
        self.generator_losses, self.discriminator_losses = [], []
        self.discriminator_acc_fake, self.discriminator_acc_real = [], []
    
    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad
        
    def setup_input(self, data):
        # Get the input data and labels
        self.inputs = data['Inputs'].to(self.device)
        self.labels = data['Labels'].to(self.device)
    
    def forward(self):
        # For each batch in the training set, forward method is called and
        # outputs stored in fake_fits variable
        self.fake_fits = self.net_G(self.inputs)
        
    def backward_D(self):
        ''' Discriminator loss takes both target and input images.
            loss_D_real is sigmoid cross-entropy loss of the target tensors and an array
            of ones. 
            loss_D_fake is sigmoid cross-entropy loss of the input tensors and an
            array of zeros.
            Discriminator loss is loss_D = loss_D_real + loss_D_fake. '''
        # Train the discriminator by feeding the fake images produced by the 
        # generator 
        fake_fits = torch.cat([self.inputs, self.fake_fits], dim=1)
        fake_preds = self.net_D(fake_fits.detach()) # detach from generator's graph so they act like constants
        # label the fake images as fake 
        self.loss_D_fake = self.GAN_loss(preds=fake_preds, target_is_real=False)
        # Now feed a batch of real images from the training set and label them as real
        real_fits = torch.cat([self.inputs, self.labels], dim=1)
        real_preds = self.net_D(real_fits)
        self.loss_D_real = self.GAN_loss(preds=real_preds, target_is_real=True)
        # Add the two losses for fake and real, take the average and call backward()
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * .5
        self.loss_D.backward()
        self.discriminator_losses += [self.loss_D.item()]
    
    def backward_G(self):
        ''' Generator loss is a sigmoid cross-entropy of input tensors and an 
            array of ones. Using the L1 loss, input tensors are structurally
            similar to the target tensors.
            Generator loss is defined as loss_G = loss_G_GAN + loss_G_L1*lambda_L1. '''
        # Train the generator by feeding the discriminator the fake fits data and 
        # fool it by assigning real labels and calculating adversarial loss.
        fake_fits = torch.cat([self.inputs, self.fake_fits], dim=1)
        fake_preds = self.net_D(fake_fits)
        self.loss_G_GAN = self.GAN_loss(preds=fake_preds, target_is_real=True)
        # Use L1 loss so tensors are not averaged over and compute the 
        # difference between the predicted channels and real channels and multiply 
        # by constant lambda 
        self.loss_G_L1 = self.L1_loss(self.fake_fits, self.labels) * self.lambda_L1
        # Add L1 loss to the adversarial loss then call backward()
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
        self.generator_losses += [self.loss_G_GAN.item()]
        
    def optimise(self):
        # Now optimise by the usual method of zeroing the gradients and calling
        # step() on the optimiser
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()
        
        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()
    
# function to log the losses and visualise the outputs from the network
class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3
    
    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count
        
def Create_Loss_Meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()
    
    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G}

def Update_Losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)
    return loss_meter 


def Loss_Plot(model, save=True):
    gen_loss = model.generator_losses
    dis_loss = model.discriminator_losses
    fig = plt.figure(figsize=(12,6))
    plt.plot(gen_loss, label='Generator Loss', color='red')
    plt.plot(dis_loss, label='Discriminator Loss', color='blue', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    if save:
        fig.savefig(f"loss_{time.time()}.png")
  
# def Accuracy_Plot(model, save=False):
#     dis_acc_fake = model.discriminator_acc_fake
#     dis_acc_real = model.discriminator_acc_real
#     fig = plt.figure(figsize=(12,6))
#     plt.plot(dis_acc_fake, label='Fake', color='red')
#     plt.plot(dis_acc_real, label='Real', color='blue', linestyle='--')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Discriminator Accuracy on Fake vs Real Predictions')
#     plt.legend()
#     plt.show()
#     if save:
#         fig.savefig(f"loss_{time.time()}.png")
    
def ConCat(inputs, labels):
    # takes a batch of images
    fits_files = torch.cat([inputs, labels], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in fits_files:
        rgb_imgs.append(img)
    return np.stack(rgb_imgs, axis=0)

def Visualise(model, data, save=False):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    fake_fits = model.fake_fits.detach()
    real_fits = model.labels
    inputs = model.inputs
    fake_fits_files = ConCat(inputs, fake_fits)
    real_fits_files = ConCat(inputs, real_fits)
    fig = plt.figure(figsize=(20,11))
    for i in range(5):
        ax = plt.subplot(3, 5, i+1)
        ax.imshow(inputs[i][0].cpu(), cmap='gray')
        ax.set_title("Input Image")
        ax.axis("off")
        ax = plt.subplot(3, 5, i+1+5)
        ax.imshow(fake_fits_files[i])
        ax.set_title("Generated ")
        ax.axis("off")
        ax = plt.subplot(3, 5, i+1+10)
        ax.imshow(real_fits_files[i])
        ax.set_title("Real Image")
        ax.axis("off")
    plt.show()
    if save:
        fig.savefig(f"colorisation_{time.time()}.png")

def Log_Results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")
        
# now train the network, display epochs and losses
def Train_Model(model, trainloader, epochs, display_every=30):
    print("Starting training....")
    start = time.time()
    data = next(iter(trainloader)) # batch for visualising the model output after fixed intervals after training
    for e in range(epochs):
        # function returning a dictionary of objects to log the losses of the complete network
        loss_meter_dict = Create_Loss_Meters() 
        i = 0
        for data in tqdm(trainloader):
            model.setup_input(data)
            model.optimise()
            Update_Losses(model, loss_meter_dict, count=data['Inputs'].size(0)) # updates the log objects
            i += 1
        print(f"\nEpoch {e+1}/{epochs}")
        if i % display_every == 0: 
            print(f"Iteration {i}/{len(trainloader)}")
        total_loss = Log_Results(loss_meter_dict) # function prints out the losses
        print(total_loss)
    Loss_Plot(model, save=True)
    #Accuracy_Plot(model, save=False)
    Visualise(model, data)
    endtime = time.time()
    end = endtime - start
    print("Time to train network: {:.2f}s".format(end))

# initialise the network
model = GANModel()
Train_Model(model, trainloader, epochs=20)




















                
        
    




