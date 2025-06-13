import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: 2.2 Your VAE model here!
class VAE(nn.Module):
    """
    This model is a VAE for MNIST, which contains an encoder and a decoder.
    
    The encoder outputs mu_phi and log (sigma_phi)^2
    The decoder outputs mu_theta
    """
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        """
        You should define your model parameters and the network architecture here.
        """
        super(VAE, self).__init__()
        
        # TODO: 2.2.1 Define your encoder and decoder
        # Encoder
        # Output the mu_phi and log (sigma_phi)^2
        raise ValueError("Not Implemented yet!")

        # Decoder
        # Output the recon_x or mu_theta
        raise ValueError("Not Implemented yet!")

    def encode(self, x):
        """ 
        Encode the image into z, representing q_phi(z|x) 
        
        Args:
            - x: the input image, we have to flatten it to (batchsize, 784) before input

        Output:
            - mu_phi, log (sigma_phi)^2
        """
        # TODO: 2.2.2 finish the encode code, input is x, output is mu_phi and log(sigma_theta)^2
        raise ValueError("Not implemented yet!")
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """ Reparameterization trick """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        """ 
        Decode z into image x

        Args:
            - z: hidden code 
            - labels: the labels of the inputs, useless here
        
        Hint: During training, z should be reparameterized! While during inference, just sample a z from random.
        """
        # TODO: 2.2.3 finish the decoding code, input is z, output is recon_x or mu_theta
        # Hint: output should be within [0, 1], maybe you can use torch.sigmoid()
        raise ValueError("Not Implemented yet!")
        return recon_x

    def forward(self, x, labels):
        """ x: shape (batchsize, 28, 28) labels are not used here"""
        # TODO: 2.2.4 passing the whole model, first encoder, then decoder, output all we need to cal loss
        # Hint1: all input data is [0, 1], 
        # and input tensor's shape is [batch_size, 1, 28, 28], 
        # maybe you have to change the shape to [batch_size, 28 * 28] if you use MLP model using view()
        # Hint2: maybe 3 or 4 lines of code is OK!
        # x = x.view(-1, 28 * 28)
        raise ValueError("Not Implemented yet!")
        return recon_x, mu, log_var

# TODO: 2.3 Calculate vae loss using input and output
def vae_loss(recon_x, x, mu, log_var, var=0.5):
    """ 
    Compute the loss of VAE

    Args:
        - recon_x: output of the Decoder, shape [batch_size, 1, 28, 28]
        - x: original input image, shape [batch_size, 1, 28, 28]
        - mu: output of encoder, represents mu_phi, shape [batch_size, latent_dim]
        - log_var: output of encoder, represents log (sigma_phi)^2, shape [batch_size, latent_dim]
        - var: variance of the decoder output, here we can set it to be a hyperparameter
    """
    # TODO: 2.3 Finish code!
    # Reconstruction loss (MSE or other recon loss)
    # KL divergence loss
    # Hint: Remember to normalize of batches, we need to cal the loss among all batches and return the mean!

    raise ValueError("Not Implemented yet!")
    return loss

# TODO: 3 Design the model to finish generation task using label
class GenModel(nn.Module):
    raise ValueError("Not Implemented yet!")