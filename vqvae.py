import torch.nn as nn
import torch

class VQVAE(nn.Module):
    
    def __init__(self, n_cb_vectors, cb_dim, color_channels):
        
        super().__init__()
        
        self.color_channels = color_channels
        self.n_cb_vectors = n_cb_vectors
        self.cb_dim = cb_dim
        
        self.codebook = nn.Embedding(num_embeddings = n_cb_vectors, embedding_dim = cb_dim)
        
        negative_slope = 0.1
        
        encoder_dims = [16, 4]
        
        self.encoder = nn.Sequential(
            nn.Conv2d(color_channels, encoder_dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(encoder_dims[0]),
            nn.LeakyReLU(negative_slope),
            nn.Conv2d(encoder_dims[0],  encoder_dims[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(encoder_dims[1]),
            nn.LeakyReLU(negative_slope),
        )
        
        # make encoder independent from codebook
        self.pre_quant_conv = nn.Conv2d(encoder_dims[1], cb_dim, kernel_size=3, stride=1, padding=1)
        
        decoder_dims = [4, 16, 32]
        self.post_quant_conv = nn.Conv2d(cb_dim, decoder_dims[0], kernel_size=3, stride=1, padding=1)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(decoder_dims[0], decoder_dims[1], kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(decoder_dims[1]),
            nn.LeakyReLU(negative_slope),
            nn.ConvTranspose2d(decoder_dims[1], decoder_dims[2], kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.BatchNorm2d(decoder_dims[2]),
            nn.LeakyReLU(negative_slope),
            # smoothen the final image
            nn.Conv2d(decoder_dims[2], color_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh() # since input is in [-1, 1]
        )
        
        # recommended by original paper
        # hyperparameter should be in range (0.1, 2.0), quite stable
        # in front of commitment loss (forces encoder to work with codebook)
        self.beta = 0.25
        
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        print(f"Loaded model from {path}")

    def save_model(self, path):
        """
        Save the model's parameters to the given path.
        """
        torch.save(self.state_dict(), path)
        print(f"Saved model to {path}")
        
    def forward(self, x):
        
        #print(x.shape)
        
        encoder_output = self.encoder(x)
        
        #print(x.shape)
        
        pre_quant = self.pre_quant_conv(encoder_output)
        
        #print(pre_quant.shape)
        
        # B, C, H, W -> B, H, W, C
        B, C, H, W = pre_quant.shape
        quant_input = pre_quant.permute(0, 2, 3, 1)
        quant_input = quant_input.reshape((B, -1, C))
        
        
        # quant_input = B, num_vectors, codebook_dim
        
        # None adds an extra dimension like unsqueeze(0)
        codebook_vectors = self.codebook.weight.unsqueeze(0).repeat((B, 1, 1))
        
        # codebook_vectors = B, vectors in codebook, codebook_dim
        
        # cdist takes l2 norm by default
        dist = torch.cdist(quant_input, codebook_vectors)
        
        # dist = B, vectors in image, vectors in codebook (for every vector in image, we get every l2 distance)
        # to each of the codebook vectors
        
        # (B, vectors in image, index of nearest codebook vector)
        min_indices = torch.argmin(dist, dim = -1)
        
        # (B * vectors in image, codebook_dim)
        quantized = torch.index_select(self.codebook.weight, dim=0, index=min_indices.view(-1))
        
        # reshape so we can compare the two
        quant_input = quant_input.reshape((-1, self.cb_dim))
        
        # codebook should be close to what the encoder spits out
        codebook_loss = torch.mean((quant_input.detach() - quantized)**2)
        
        # encoder should spit out what the codebook already offers
        commitment_loss = torch.mean((quantized.detach() - quant_input)**2)
        
        quantized_loss = codebook_loss + self.beta * commitment_loss
        
        # straight through gradient (argmax blocks the gradient so we copy them)
        # = quantized
        quantized = quant_input + (quantized - quant_input).detach()
        quantized = quantized.view((B, H, W, -1))
        quantized = quantized.permute(0, 3, 1, 2)
        
        post_quant = self.post_quant_conv(quantized)
        
        #print(post_quant.shape)
        
        decoder_output = self.decoder(post_quant)
        
        return decoder_output, quantized_loss
    
    def get_indices(self, x):
        
        with torch.no_grad():
        
            encoder_output = self.encoder(x)

            #print(x.shape)

            pre_quant = self.pre_quant_conv(encoder_output)

            #print(pre_quant.shape)

            # B, C, H, W -> B, H, W, C
            B, C, H, W = pre_quant.shape
            quant_input = pre_quant.permute(0, 2, 3, 1)
            quant_input = quant_input.reshape((B, -1, C))


            # quant_input = B, num_vectors, codebook_dim

            # None adds an extra dimension like unsqueeze(0)
            codebook_vectors = self.codebook.weight.unsqueeze(0).repeat((B, 1, 1))

            # codebook_vectors = B, vectors in codebook, codebook_dim

            # cdist takes l2 norm by default
            dist = torch.cdist(quant_input, codebook_vectors)

            # dist = B, vectors in image, vectors in codebook (for every vector in image, we get every l2 distance)
            # to each of the codebook vectors

            # (B, vectors in image, index of nearest codebook vector)
            min_indices = torch.argmin(dist, dim = -1)
        
        return min_indices
    
    def get_image_from_indices(self, min_indices, H, W):
        
        B = min_indices.shape[0]
        
        with torch.no_grad():
        
            # (B * vectors in image, codebook_dim)
            quantized = torch.index_select(self.codebook.weight, dim=0, index=min_indices.view(-1))

            # straight through gradient (argmax blocks the gradient so we copy them)
            # = quantized
            quantized = quantized.view((B, H, W, -1))
            quantized = quantized.permute(0, 3, 1, 2)

            post_quant = self.post_quant_conv(quantized)

            #print(post_quant.shape)

            decoder_output = self.decoder(post_quant)
        
        return decoder_output