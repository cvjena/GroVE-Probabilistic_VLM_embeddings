
import torch.nn as nn
import os
import torch
import gpytorch
from tqdm import tqdm
import torch.nn.functional as F
from gpytorch.mlls import VariationalELBO

import utils
from config import get_cfg_defaults 
import model



cfg = get_cfg_defaults()


os.makedirs(cfg.MODEL.CHECKPOINT_DIR, exist_ok=True)

# Load dataset
data, loaders = utils.load_data_loader(cfg.DATASET.NAME, os.path.join(cfg.DATASET.PATH, cfg.DATASET.NAME))
train_dataset, val_dataset, test_dataset = data['train'], data['val'], data['test']
train_loader, val_loader, test_loader = loaders['train'], loaders['val'], loaders['test']
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load VLM
VLM_Net = utils.load_vlm(device=device, model_name = cfg.MODEL.VLM, backbone = cfg.MODEL.VLM_BACKBONE, model_path=cfg.MODEL.VLM_WEIGHTS)  
VLM_Net.eval()


# GroVE model

# A random set of low dimensional latent points
X = torch.randn(train_dataset.__len__(), cfg.MODEL.LATENT_DIM, requires_grad=True, device=device)

# Gaussian Process models 
model_image = model.LVM(data_dim=cfg.MODEL.EMB_DIM, latent_dim=cfg.MODEL.LATENT_DIM, n_inducing=cfg.MODEL.NUM_INDUCING_PTS).to(device)
model_text = model.LVM(data_dim=cfg.MODEL.EMB_DIM, latent_dim=cfg.MODEL.LATENT_DIM, n_inducing=cfg.MODEL.NUM_INDUCING_PTS).to(device)

likelihood_text = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=cfg.MODEL.EMB_DIM).to(device)
likelihood_image = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=cfg.MODEL.EMB_DIM).to(device)

optimizer = torch.optim.Adam([
    {'params': model_text.parameters()},
    {'params': model_image.parameters()},
    {'params': likelihood_text.parameters()},
    {'params': likelihood_image.parameters()},
    {'params': [X]}
    ], lr=cfg.TRAIN.LEARNING_RATE)

    # Variational ELBO for both GPs
mll_text = VariationalELBO(likelihood_text, model_text, num_data=train_dataset.__len__())
mll_image = VariationalELBO(likelihood_image, model_image, num_data=train_dataset.__len__())

score = 1e8


for epoch in range(cfg.TRAIN.NUM_EPOCHS):
    with tqdm(train_loader, unit='batch') as tepoch:
        for (idx, batch) in enumerate(tepoch):
            tepoch.set_description('Epoch {}'.format(epoch))
            xI, xT  = batch[0].to(device), batch[1].to(device)
            batch_index = batch[5]

            # Get the CLIP embeddings
            with torch.no_grad():
                xfI, xfT = VLM_Net(xI, xT)
            xfI = F.normalize(xfI, p=2, dim=1)
            xfT = F.normalize(xfT, p=2, dim=1)

            # Latent variables to be optimized
            X_batch = X[batch_index]

            output_text = model_text(X_batch)
            output_image = model_image(X_batch)

            # Compute ELBO for both models

            #print(output_text.mean, output_text.variance)
            loss_text = -mll_text(output_text, xfT)
            loss_image = -mll_image(output_image, xfI)

            kl_loss_im2txt = utils.kl_divergence_gaussians(output_image.mean, output_image.variance, output_text.mean, output_text.variance).mean()
            kl_loss_txt2im = utils.kl_divergence_gaussians(output_text.mean, output_text.variance, output_image.mean, output_image.variance).mean()
            #kl_loss_im2txt = 

            #print(kl_loss_im2txt)
            #print(loss_text, loss_image, kl_loss_im2txt, kl_loss_txt2im)
            total_loss = (loss_text + loss_image)/100 + cfg.TRAIN.LOSS_WEIGHT*(kl_loss_im2txt + kl_loss_txt2im)
            total_loss.backward()
            optimizer.step()
 
        torch.save(model_image.state_dict(), os.path.join(cfg.MODEL.CHECKPOINT_DIR,'model_image_last.pth'))
        torch.save(model_text.state_dict(), os.path.join(cfg.MODEL.CHECKPOINT_DIR,'model_text_last.pth'))
        torch.save(likelihood_text.state_dict(), os.path.join(cfg.MODEL.CHECKPOINT_DIR,'likelihood_text_last.pth'))
        torch.save(likelihood_image.state_dict(), os.path.join(cfg.MODEL.CHECKPOINT_DIR,'likelihood_image_last.pth'))
        torch.save(X, os.path.join(cfg.MODEL.CHECKPOINT_DIR,'X_last.pth'))
        print(f"Epoch [{epoch}/{cfg.TRAIN.NUM_EPOCHS}], Step [{idx}/{tepoch}], Loss: {total_loss.item()}")

    if total_loss.item() < score:
        score = total_loss.item()
        torch.save(model_image.state_dict(), os.path.join(cfg.MODEL.CHECKPOINT_DIR,'model_image_best.pth'))
        torch.save(model_text.state_dict(), os.path.join(cfg.MODEL.CHECKPOINT_DIR,'model_text_best.pth'))
        torch.save(likelihood_text.state_dict(), os.path.join(cfg.MODEL.CHECKPOINT_DIR,'likelihood_text_best.pth'))
        torch.save(likelihood_image.state_dict(), os.path.join(cfg.MODEL.CHECKPOINT_DIR,'likelihood_image_best.pth'))
        torch.save(X, os.path.join(cfg.MODEL.CHECKPOINT_DIR,'X_best.pth'))      
