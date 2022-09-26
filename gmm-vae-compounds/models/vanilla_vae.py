from .modules import *
from scipy.stats import pearsonr

# Standard VAE
class VanillaVAE(pl.LightningModule):
    def __init__(self, encoder_layers, input_decoder_layers, guiding_decoder_layers,
                 var_transformation=lambda x: torch.exp(x) ** 0.5, learning_rate=0.001,
                 loss_function_weights=(1., 1., 1., 1., 0.), batch_norm=False, optimizer="adam",
                 encoder_dropout_rate=0, decoders_dropout_rate=0, clip_guiding_rec=False, guiding_clip_min=0, guiding_clip_max=100):
        
        super().__init__()
        # Establish encoder
        if len(encoder_layers) == 4:
            if batch_norm:
                self.encoder = EncoderTwoLayersBatchNormConfigurable(encoder_layers, var_transformation)
            else:
                self.encoder = EncoderTwoLayersConfigurable(encoder_layers, var_transformation)

        if len(encoder_layers) == 5:
            if batch_norm:
                self.encoder = EncoderThreeLayersBatchNormConfigurable(encoder_layers, var_transformation)
            else:
                self.encoder = EncoderThreeLayersConfigurable(encoder_layers, var_transformation,
                                                              dropout_rate=encoder_dropout_rate)

        # Establish input decoder
        if len(input_decoder_layers) == 4:
            if batch_norm:
                self.input_decoder = DecoderTwoLayersBatchNormConfigurable(input_decoder_layers)
            else:
                self.input_decoder = DecoderTwoLayersConfigurable(input_decoder_layers)

        if len(input_decoder_layers) == 5:
            if batch_norm:
                self.input_decoder = DecoderThreeLayersBatchNormConfigurable(input_decoder_layers)
            else:
                self.input_decoder = DecoderThreeLayersConfigurable(input_decoder_layers, dropout_rate=decoders_dropout_rate)

        # Establish guiding decoder
        if len(guiding_decoder_layers) == 4:
            if batch_norm:
                self.guiding_decoder = DecoderTwoLayersBatchNormConfigurable(guiding_decoder_layers)
            else:
                self.guiding_decoder = DecoderTwoLayersConfigurable(guiding_decoder_layers)
        if len(guiding_decoder_layers) == 5:
            if batch_norm:
                self.guiding_decoder = DecoderThreeLayersBatchNormConfigurable(guiding_decoder_layers)
            else:
                self.guiding_decoder = DecoderThreeLayersConfigurable(guiding_decoder_layers, dropout_rate=decoders_dropout_rate)

        self.latent_dim = encoder_layers[-1]
        self.learning_rate = learning_rate

        # Loss function weights
        self.input_rec_likelihood_weight = loss_function_weights[0]
        self.guiding_rec_likelihood_weight = loss_function_weights[1]
        self.entropy_weight = loss_function_weights[2]
        self.prior_likelihood_weight = loss_function_weights[3]
        self.l2_reg_weight = loss_function_weights[4]
        
        self.clip_guiding_rec = clip_guiding_rec
        self.guiding_clip_min = guiding_clip_min
        self.guiding_clip_max = guiding_clip_max

        self.optimizer = optimizer

    def forward(self, inputs):
        z_means, z_stds = self.encoder(inputs)

        normal_rv = self.make_normal_rv(z_means, z_stds)
        #normal_rv = td.normal.Normal(mu, z_vars.pow(0.5))
        z_sample = normal_rv.rsample()
        
        prior = self.make_standard_normal_prior()

        # Input data decoder
        input_rec = self.input_decoder(z_sample)
        # Guiding data decoder
        guiding_rec = self.guiding_decoder(z_sample)
        if self.clip_guiding_rec:
            guiding_rec = torch.clip(guiding_rec, min=self.guiding_clip_min, max=self.guiding_clip_max)

        return input_rec, guiding_rec, z_sample, normal_rv, z_means, prior

    def loss_function(self, input_true, input_rec, guiding_true, guiding_rec, z_sample, normal_rv, prior):   # TO DO: maybe change to accepting packed vae output
        neg_input_rec_likelihood = F.mse_loss(input_true, input_rec)
        neg_guding_rec_likelihood = self.mse_loss_with_nans(guiding_rec, guiding_true)
        entropy = torch.mean(normal_rv.entropy())   # Calculated on MultiVariateNormal

        # Calculate likelihood of z_sample w.r.t prior
        #prior = self.make_standard_normal_prior()
        prior_likelihood = torch.mean(prior.log_prob(z_sample))

        return neg_input_rec_likelihood, neg_guding_rec_likelihood, entropy, prior_likelihood
    
    def configure_optimizers(self):
        if self.optimizer == "adam":
            optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg_weight)
        if self.optimizer == "sgd":
            optim = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg_weight)
        return optim


    def training_step(self, train_batch, batch_idx):
        input_X, guiding_X= train_batch[0], train_batch[1]    # Designed to work with data loader outputting also class labels, although they are nor used
        input_X, guiding_X = input_X.float(), guiding_X.float()
        # Forward pass
        input_rec, guiding_rec, z_sample, normal_rv, _, prior = self.forward(input_X)
        # Calculate loss
        neg_input_rec_likelihood, neg_guiding_rec_likelihood, entropy, prior_likelihood = self.loss_function(input_X, input_rec, guiding_X, guiding_rec,
                                                                           z_sample, normal_rv, prior)

        loss = 1. * neg_input_rec_likelihood + 1.* neg_guiding_rec_likelihood - 1. * entropy -1. * prior_likelihood

        try:
            self.log("input_rec_likelihood", neg_input_rec_likelihood, on_step=False, on_epoch=True, prog_bar=True)
            self.log("guiding_rec_likelihood", neg_guiding_rec_likelihood, on_step=False, on_epoch=True, prog_bar=True)
            self.log("entropy", entropy, on_step=False, on_epoch=True, prog_bar=True)
            self.log("gmm_likelihood", gmm_likelihood, on_step=False, on_epoch=True, prog_bar=True)
            self.log("loss", loss, on_step=False, on_epoch=True)
            
            return loss
        except:
            return loss

    def validation_step(self, val_batch, batch_idx):
        input_X, guiding_X = val_batch[0], val_batch[1]
        input_X, guiding_X = input_X.float(), guiding_X.float()
        input_rec, guiding_rec, z_sample, normal_rv, gmm = self.forward(input_X)

        input_mse = F.mse_loss(input_X, input_rec)
        guiding_mse = F.mse_loss(guiding_X, guiding_rec)

        return {"input_mse": input_mse, "guiding_mse": guiding_mse}

    def validation_epoch_end(self, outputs):
        avg_guiding_mse = torch.stack([x["guiding_mse"] for x in outputs]).mean()
        avg_guiding_rmse = avg_guiding_mse ** 0.5

        avg_input_mse = torch.stack([x["input_mse"] for x in outputs]).mean()
        avg_input_rmse = avg_input_mse ** 0.5

        self.log("guiding_val_mse", avg_guiding_mse)
        self.log("guiding_val_rmse", avg_guiding_rmse)
        self.log("input_val_mse", avg_input_mse)
        self.log("input_val_rmse", avg_input_rmse)
    
    def make_standard_normal_prior(self):
        # Make standard normal prior
        return td.MultivariateNormal(torch.zeros(self.latent_dim).to(self.device), torch.eye(self.latent_dim).to(self.device))

    @staticmethod
    def make_normal_rv(means, vars):
        return td.MultivariateNormal(means, torch.stack([torch.diag(vars[i, :]) for i in range(vars.shape[0])], axis=0))
    
    def mse_loss_with_nans(self, rec, target):
        # When missing data are nan's
        mask = torch.isnan(target)
        neg_likelihood = F.mse_loss(rec[~mask], target[~mask])
        if torch.isnan(neg_likelihood):
            return torch.tensor(0.0).to(self.device)
        else:
            return neg_likelihood