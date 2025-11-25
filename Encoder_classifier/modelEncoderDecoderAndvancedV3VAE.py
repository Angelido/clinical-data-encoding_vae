import torch
import torch.nn as nn
#from torcheval.metrics.functional import r2_score

class MIEOVAE(nn.Module):
	def __init__(self, latent_dim:int=32, input_dim:int=100, hidden_dims:list=[128, 64], binary:int=None):
		'''
		MIOE VAE model
		A variational autoencoder that handles continuous and binary data
		Args:
			latent_dim (int): Dimension of the latent space
			input_dim (int): Number of input features
			hidden_dims (list): List of hidden dimensions for the encoder (the decoder is symmetric)
			binary (int): Number of binary featueres at the end of the input (if any)
		'''
		super().__init__()
		self.binary_features = binary
		self.latent_dim = latent_dim

		# --- Encoder ---
		self.encoder = nn.Sequential(
			nn.Linear(in_features=input_dim, out_features=hidden_dims[0]),
			nn.Tanh(),
		)
		for i in range(len(hidden_dims) - 1):
			self.encoder.append(nn.Linear(in_features=hidden_dims[i], out_features=hidden_dims[i+1]))
			self.encoder.append(nn.Tanh())
		
		# --- mu - logvar ---
		self.mu = nn.Linear(in_features=hidden_dims[-1], out_features=latent_dim)
		self.logvar = nn.Linear(in_features=hidden_dims[-1], out_features=latent_dim)

		# --- Decoder ---
		self.decoder = nn.Sequential(
			nn.Linear(in_features=latent_dim, out_features=hidden_dims[-1]),
			nn.Tanh(),
		)
		for i in range(len(hidden_dims) - 1):
			self.decoder.append(nn.Linear(in_features=hidden_dims[-i - 1], out_features=hidden_dims[-i -2]))
			self.decoder.append(nn.Tanh())
		
		self.binary_dec = nn.Sequential(
			nn.Linear(in_features=hidden_dims[0], out_features=binary),
			nn.Sigmoid(),
		)

		self.continuous_dec = nn.Linear(in_features=hidden_dims[0], out_features=input_dim - binary)
		
	def reparameterize(self, mu:torch.Tensor, logvar:torch.Tensor) -> torch.Tensor:
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		return mu + eps * std
	
	def encode(self, x:torch.Tensor) -> tuple:
		encoded = self.encoder(x)
		mu = self.mu(encoded)
		logvar = self.logvar(encoded)
		return mu, logvar

	def forward(self, x:torch.Tensor) -> tuple:
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		reconstructed = self.decoder(z)
		binary_rec = self.binary_dec(reconstructed)
		continuous_rec = self.continuous_dec(reconstructed)
		reconstructed = torch.cat((continuous_rec, binary_rec), dim=1)
		return reconstructed, mu, logvar
	
	def kl(self, mu:torch.Tensor, logvar:torch.Tensor) -> torch.Tensor:
		# KL divergence between N(mu, var) and N(0, 1)
		return (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)).mean()
	
	def get_loss_masked(self, x:torch.Tensor, mask_percentage:float=0.1, beta:float=1.0, binary_weight:float=1, null_mask:torch.Tensor=None) -> tuple:
		# Create a mask
		mask = torch.rand_like(x) > mask_percentage
		x_masked = x * mask.float()
		x_masked = x_masked - (~mask).float()

		reconstructed, mu, logvar = self.forward(x_masked)
		cont_dim = x.shape[1] - self.binary_features
		mse_loss = nn.functional.mse_loss(reconstructed[:, :cont_dim]*null_mask[:, :cont_dim], x[:, :cont_dim]*null_mask[:, :cont_dim], reduction='mean')
		bce_loss = nn.functional.mse_loss(reconstructed[:, cont_dim:]*null_mask[:, cont_dim:], x[:, cont_dim:]*null_mask[:, cont_dim:], reduction='mean')
		reconstruction_loss = mse_loss + binary_weight * bce_loss

		kl_loss = self.kl(mu, logvar)
		total_loss = reconstruction_loss + beta * kl_loss
		return total_loss, reconstruction_loss, kl_loss
	
	def train_epoch(self, dataloader, optimizer, beta:float=1.0, mask_percentage:float=0.1, binary_weight:float=1):
		self.train()
		losses = {
			'total_loss': 0.0,
			'reconstruction_loss': 0.0,
			'kl_loss': 0.0
		}
		for x, null_mask in dataloader:
			optimizer.zero_grad()
			total_loss, reconstruction_loss, kl_loss = self.get_loss_masked(x, mask_percentage, beta, binary_weight, null_mask)
			total_loss.backward()
			optimizer.step()

			losses['total_loss'] += total_loss.item()
			losses['reconstruction_loss'] += reconstruction_loss.item()
			losses['kl_loss'] += kl_loss.item()

		num_batches = len(dataloader)
		for key in losses:
			losses[key] /= num_batches
		return losses
	
	def eval_epoch(self, dataloader, beta:float=1.0, mask_percentage:float=0.1, binary_weight:float=1):
		self.eval()
		losses = {
			'total_loss': 0.0,
			'reconstruction_loss': 0.0,
			'kl_loss': 0.0
		}
		with torch.no_grad():
			for x, null_mask in dataloader:
				total_loss, reconstruction_loss, kl_loss = self.get_loss_masked(x, mask_percentage, beta, binary_weight, null_mask)

				losses['total_loss'] += total_loss.item()
				losses['reconstruction_loss'] += reconstruction_loss.item()
				losses['kl_loss'] += kl_loss.item()
		num_batches = len(dataloader)
		for key in losses:
			losses[key] /= num_batches
		return losses
	
	def fit(self, tr, vl, optim, bs, bw, kl_beta, ep, mask_perc, es, pedantic=False):
		"""
		Train the VAE model
		Args:
			tr (torch.utils.data.Dataloader): Training dataloader
			vl (torch.utils.data.Dataloader): Validation dataloader
			optim (torch.optim.Optimizer): Optimizer
			bs (int): Batch size
			bw (float): Binary weight for loss
			kl_beta (float): Beta for KL divergence (incremented from 0 to kl_beta during training)
			ep (int): Number of epochs
			mask_perc (float): Masking percentage
			es (int): Early stopping patience
			pedantic (bool): Whether to use pedantic mode (print every epoch)
		"""
		best_val_loss = float('inf')
		epochs_no_improve = 0
		current_kl_beta = 0.0

		for epoch in range(ep):
			# Training step
			train_losses = self.train_epoch(tr, optim, beta=current_kl_beta, mask_percentage=mask_perc, binary_weight=bw)
			val_losses = self.eval_epoch(vl, beta=current_kl_beta, mask_percentage=mask_perc, binary_weight=bw)
			if pedantic:
				print(f'Epoch {epoch+1}/{ep} | Train Loss: {train_losses["total_loss"]:.4f} | Val Loss: {val_losses["total_loss"]:.4f}')
			# Early stopping
			if val_losses['total_loss'] < best_val_loss:
				best_val_loss = val_losses['total_loss']
				epochs_no_improve = 0
				best_model_state = self.state_dict()
			else:
				epochs_no_improve += 1
				if epochs_no_improve >= es:
					print(f'Early stopping at epoch {epoch+1}')
					self.load_state_dict(best_model_state)
					break
			# Increment KL beta
			if current_kl_beta < kl_beta:
				current_kl_beta += kl_beta / (ep // 2)
				if current_kl_beta > kl_beta:
					current_kl_beta = kl_beta

		print(f'Training complete. Best Val Loss: {best_val_loss:.4f}')

	def saveModel(self, path:str):
		"""Save the model to a file"""
		torch.save(self, path)

	def freeze(self):
		"""Freeze the model parameters"""
		for param in self.parameters():
			param.requires_grad = False

if __name__ == "__main__":
	model = MIEOVAE(hidden_dims=[60, 40, 30, 20, 3])
	print(model)