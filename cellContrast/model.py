import torch
from torch import nn
import torch.nn.functional as F

# Set the random seed for PyTorch and NumPy
torch.manual_seed(0)


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):

        batch_size = embeddings.shape[0]
        
        embeddings = F.normalize(embeddings, p=2, dim=1)
        sim_matrix = torch.exp(torch.matmul(embeddings, embeddings.T) / self.temperature)
       
        pos_mask = (labels == 1).type(torch.bool)

        neg_mask = ~pos_mask
        neg_mask.fill_diagonal_(False)  # remove diagonal items

        pos_similarities = (sim_matrix * pos_mask).sum(dim=1)
        neg_similarities = (sim_matrix * neg_mask).sum(dim=1)
        
        loss = -torch.mean(torch.log(pos_similarities / neg_similarities))

        return loss

class ProjectionHead(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    
    
class Encoder(nn.Module):
    def __init__(self, n_input, n_latent, n_layers=2, n_hidden=1024, dropout_rate=0,negative_slope=0.01):
        
        super().__init__()

        self.n_input = n_input
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.dropout_rate = dropout_rate
        self.negative_slope = negative_slope

        # Define the hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(self.n_layers - 1):
            if(i==0):
                self.hidden_layers.append(nn.Linear(self.n_input, self.n_hidden))
            else:
                self.hidden_layers.append(nn.Linear(self.n_hidden, self.n_hidden))
            self.hidden_layers.append(nn.BatchNorm1d(self.n_hidden))
            self.hidden_layers.append(nn.LeakyReLU(self.negative_slope))
            self.hidden_layers.append(nn.Dropout(self.dropout_rate))

        # Define the output layer
        self.output_layer = nn.Linear(self.n_hidden, self.n_latent)
        
        # Define additional layers after the output layer
        self.bn_layer = nn.BatchNorm1d(self.n_latent)
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.activation_layer = nn.LeakyReLU(self.negative_slope)

    def forward(self, x):
        
        # Pass input through the hidden layers
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)

        
        # Compute the representations
        embeddings = self.output_layer(x)

        embeddings = self.bn_layer(embeddings)
        embeddings = self.activation_layer(embeddings)
        embeddings = self.dropout_layer(embeddings)
        

        return embeddings

class CellContrastModel(nn.Module):


    def __init__(self,n_input, n_encoder_hidden=1024,n_encoder_latent=512,n_encoder_layers=2,\
                 n_projection_hidden=256,n_projection_output=128,dropout_rate=0,negative_slope=0.01):

        super(CellContrastModel,self).__init__()
     
        self.encoder = Encoder(n_input, n_encoder_latent, n_encoder_layers, n_encoder_hidden, dropout_rate,negative_slope)
        
        self.projection = ProjectionHead(n_encoder_latent, n_projection_hidden, n_projection_output)

        

    def forward(self,x):
        
        representation = self.encoder(x)
        projection = self.projection(representation)
        
        return representation, projection
    
    
if __name__ == '__main__':
    
    model = CellContrastModel(n_input=351, n_encoder_hidden=1024,n_encoder_latent=512,n_encoder_layers=2,n_projection_hidden=256,n_projection_output=128)
    print(model)
    
    pass

