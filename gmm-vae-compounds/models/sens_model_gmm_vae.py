from .modules import *
from scipy.stats import pearsonr

from .gmm_vae import GMMVAE

# Model with GMM VAE
class SensitivityModelGMMVAE(pl.LightningModule):
    """
    Class implementing multi-task generative model for drug sensitivity prediction, with 
    GMM VAE guided by guiding data as generative module.
    """
    
    def __init__(self, drug_model, cell_line_model, sensitivity_prediction_network, learning_rate=0.001, optimizer="adam",
              aen_reconstruction_loss_weight=1., sensitivity_loss_weight=1., vae_dataloader=None, vae_training_num_epochs=100, vae_training_step_rate=1000):
        super().__init__()

        self.drug_model = drug_model
        self.cell_line_model = cell_line_model
        self.sensitivity_prediction_network = sensitivity_prediction_network

        self.learning_rate = learning_rate
        self.optimizer = optimizer

        self.aen_reconstruction_loss_weight = aen_reconstruction_loss_weight
        self.sensitivity_loss_weight = sensitivity_loss_weight
        
        self.vae_dataloader = vae_dataloader
        self.vae_training_num_epochs = vae_training_num_epochs
        self.vae_training_step_rate = vae_training_step_rate
        self.total_step = 0

    def forward(self, drug_input, cell_line_input):
        # Drug model part
        vae_out = self.drug_model(drug_input)
        input_rec, guiding_rec, z_sample, normal_rv, gmm, z_means = vae_out   # TO DO: maybe changed
        # Cell line model part
        aen_out = self.cell_line_model(cell_line_input)
        rec, latent = aen_out

        # Concatenate latent representations
        ff_input = torch.cat((z_means, latent), axis=1)

        # Forward pass of feed forward network
        sensitivity_pred = self.sensitivity_prediction_network(ff_input)

        return vae_out, aen_out, sensitivity_pred

    def loss_function(self, batch_data, vae_out, aen_out, sensitivity_pred):
        # Unpack batch data
        batch_vae_input_X, batch_vae_guiding_X, batch_vae_guiding_classes, batch_aen_input_X, batch_targets, _, _ = batch_data

        # Unpack VAE output
        #     input_rec, guiding_rec, z_sample, normal_rv, z_means = vae_out
        input_rec, guiding_rec, z_sample, normal_rv, gmm, z_means = vae_out

        # Unpack AEN output
        rec, latent = aen_out

        # Compute VAE loss
        vae_loss = self.drug_model.loss_function(batch_vae_input_X, input_rec, batch_vae_guiding_X, guiding_rec, z_sample, normal_rv, batch_vae_guiding_classes, gmm)

        # Compute AEN loss
        aen_loss = self.cell_line_model.loss_function(batch_aen_input_X, rec)

        # Compute sensitivity loss
        sensitivity_loss = self.sensitivity_prediction_network.loss_function(sensitivity_pred, batch_targets)

        return vae_loss, aen_loss, sensitivity_loss

    def training_step(self, train_batch, batch_idx):
        if self.vae_dataloader is not None:
            if self.total_step % self.vae_training_step_rate == 0:
                self.train_vae()
                
        # Unpack batch data
        batch_vae_input_X, batch_vae_guiding_X, batch_vae_guiding_classes, batch_aen_input_X, batch_targets, drug_ids, cell_line_ids = train_batch   # SUSPECTLIBLE TO CHANGES

        batch_vae_input_X = batch_vae_input_X.float()
        batch_vae_guiding_X = batch_vae_guiding_X.float()
        batch_vae_guiding_classes = batch_vae_guiding_classes.float()
        batch_aen_input_X = batch_aen_input_X.float()  
        batch_targets = batch_targets.float()   # TO DO: get rid of that

        train_batch = batch_vae_input_X, batch_vae_guiding_X, batch_vae_guiding_classes, batch_aen_input_X, batch_targets, drug_ids, cell_line_ids

        # Forward pass of the model
        vae_out, aen_out, sensitivity_pred = self.forward(batch_vae_input_X, batch_aen_input_X)

        # Calculate loss
        vae_losses, aen_loss, sensitivity_loss = self.loss_function(train_batch, vae_out, aen_out, sensitivity_pred)

        # Unpack VAE loss
        neg_input_rec_likelihood, neg_guiding_rec_likelihood, entropy, gmm_likelihood = vae_losses

        # Gather losses
        vae_loss = self.drug_model.input_rec_likelihood_weight * neg_input_rec_likelihood + self.drug_model.guiding_rec_likelihood_weight * neg_guiding_rec_likelihood - self.drug_model.entropy_weight * entropy - self.drug_model.gmm_likelihood_weight * gmm_likelihood

        # Overal loss
        loss = vae_loss + self.aen_reconstruction_loss_weight * aen_loss + self.sensitivity_loss_weight * sensitivity_loss

        # Log losses
        self.log("vae_loss", vae_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("aen_reconstruction", aen_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("sensitivity_loss", sensitivity_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.log("input_rec_likelihood", neg_input_rec_likelihood, on_step=False, on_epoch=True, prog_bar=True)
        self.log("guiding_rec_likelihood", neg_guiding_rec_likelihood, on_step=False, on_epoch=True, prog_bar=True)
        self.log("entropy", entropy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("gmm_likelihood", gmm_likelihood, on_step=False, on_epoch=True, prog_bar=True)
        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.total_step += 1

        return loss
    
    def train_vae(self):
        optimizer = self.drug_model.configure_optimizers()
        for epoch in range(self.vae_training_num_epochs):
            for batch in self.vae_dataloader:
                loss = self.drug_model.training_step(batch, 0)
                # clear gradients
                optimizer.zero_grad()

                # backward
                loss.backward()

                # update parameters
                optimizer.step()
        
    def validation_step(self, val_batch, batch_idx):
        batch_vae_input_X, batch_vae_guiding_X, batch_vae_guiding_classes, batch_aen_input_X, batch_targets, drug_ids, cell_line_ids = val_batch

        batch_vae_input_X = batch_vae_input_X.float()
        batch_vae_guiding_X = batch_vae_guiding_X.float()
        batch_vae_guiding_classes = batch_vae_guiding_classes.float()
        batch_aen_input_X = batch_aen_input_X.float()
        batch_targets = batch_targets.float()

        # Forward pass of the model
        vae_out, aen_out, sensitivity_pred = self.forward(batch_vae_input_X, batch_aen_input_X)

        # Unpack drug model output
        #         input_rec, guiding_rec, z_sample, normal_rv, z_means = vae_out
        input_rec, guiding_rec = vae_out[0], vae_out[1]
        # Unpack drug model output
        cell_line_rec, cell_line_latent = aen_out
        
        drug_input_rec_mse = F.mse_loss(batch_vae_input_X, input_rec)
        drug_guiding_rec_mse = self.drug_model.mse_loss_with_nans(guiding_rec, batch_vae_guiding_X)
        cl_input_rec_mse = F.mse_loss(batch_aen_input_X, cell_line_rec)
        sensitivity_pred_mse = F.mse_loss(batch_targets, sensitivity_pred)
        sensitivity_pred_corr = torch.corrcoef(torch.stack((batch_targets.view(-1), sensitivity_pred.view(-1))))[0, 1]

        return {"drug_input_rec_mse": drug_input_rec_mse, "drug_guiding_rec_mse": drug_guiding_rec_mse, "cl_input_rec_mse": cl_input_rec_mse,
               "sensitivity_pred_mse": sensitivity_pred_mse, "sensitivity_pred_corr": sensitivity_pred_corr}

    def validation_epoch_end(self, outputs):
        avg_drug_input_rec_mse = torch.stack([x["drug_input_rec_mse"] for x in outputs]).mean()
        avg_drug_input_rec_rmse = avg_drug_input_rec_mse ** 0.5

        avg_drug_guiding_rec_mse = torch.stack([x["drug_guiding_rec_mse"] for x in outputs]).mean()
        avg_drug_guiding_rec_rmse = avg_drug_guiding_rec_mse ** 0.5

        avg_cl_input_rec_mse = torch.stack([x["cl_input_rec_mse"] for x in outputs]).mean()
        avg_cl_input_rec_rmse = avg_cl_input_rec_mse ** 0.5

        avg_sensitivity_pred_mse = torch.stack([x["sensitivity_pred_mse"] for x in outputs]).mean()
        avg_sensitivity_pred_rmse = avg_sensitivity_pred_mse ** 0.5
        
        avg_sensitivity_pred_corr = torch.stack([x["sensitivity_pred_corr"] for x in outputs]).mean()

        self.log("val_drug_input_rec_rmse", avg_drug_input_rec_rmse)
        self.log("val_drug_guiding_rec_rmse", avg_drug_guiding_rec_rmse)
        self.log("val_cl_input_rec_rmse", avg_cl_input_rec_rmse)
        self.log("val_sensitivity_pred_rmse", avg_sensitivity_pred_rmse)
        self.log("val_sensitivity_pred_corr", avg_sensitivity_pred_corr)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optim