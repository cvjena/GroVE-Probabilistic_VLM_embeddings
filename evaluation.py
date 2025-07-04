import torch.nn as nn
import os
import torch
import gpytorch
from tqdm import tqdm
import torch.nn.functional as F
from gpytorch.mlls import VariationalELBO


from config import get_cfg_defaults 
import model
import utils



cfg = get_cfg_defaults()

# Load dataset
data, loaders = utils.load_data_loader(cfg.DATASET.NAME, os.path.join(cfg.DATASET.PATH, cfg.DATASET.NAME))
train_dataset, val_dataset, test_dataset = data['train'], data['val'], data['test']
train_loader, val_loader, test_loader = loaders['train'], loaders['val'], loaders['test']
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load VLM
VLM_Net = utils.load_vlm(device=device, model_name = cfg.MODEL.VLM, backbone = cfg.MODEL.VLM_BACKBONE, model_path=cfg.MODEL.VLM_WEIGHTS)  
VLM_Net.eval()

# Gaussian Process models 
model_image = model.LVM(data_dim=cfg.MODEL.EMB_DIM, latent_dim=cfg.MODEL.LATENT_DIM, n_inducing=cfg.MODEL.NUM_INDUCING_PTS).to(device)
model_text = model.LVM(data_dim=cfg.MODEL.EMB_DIM, latent_dim=cfg.MODEL.LATENT_DIM, n_inducing=cfg.MODEL.NUM_INDUCING_PTS).to(device)

likelihood_text = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=cfg.MODEL.EMB_DIM).to(device)
likelihood_image = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=cfg.MODEL.EMB_DIM).to(device)

model_image.load_state_dict(torch.load(os.path.join(cfg.MODEL.CHECKPOINT_DIR,'model_image_best.pth')))
model_image.eval()
model_text.load_state_dict(torch.load(os.path.join(cfg.MODEL.CHECKPOINT_DIR,'model_text_best.pth')))
model_text.eval()

likelihood_image.load_state_dict(torch.load(os.path.join(cfg.MODEL.CHECKPOINT_DIR,'likelihood_image_best.pth')))
likelihood_image.eval()
likelihood_text.load_state_dict(torch.load(os.path.join(cfg.MODEL.CHECKPOINT_DIR,'likelihood_text_best.pth')))
likelihood_text.eval()

gt_labels = []
clip_img_embeddings = []
clip_txt_embeddings = []
learned_img_embeddings_mean = []
learned_txt_embeddings_mean = []
learned_img_embeddings_variance = []
learned_txt_embeddings_variance = []

r_dict= {
    'i_u': [],
    't_u': [],
    }  

with tqdm(test_loader, unit='batch') as tepoch:
    for (idx, batch) in enumerate(tepoch):

        xI, xT, label  = batch[0].to(device), batch[1].to(device), batch[2]
        gt_labels.extend(label)

        # Get the CLIP embeddings
        with torch.no_grad():
            xfI, xfT = VLM_Net(xI, xT)

        xfI = F.normalize(xfI, p=2, dim=1)
        xfT = F.normalize(xfT, p=2, dim=1)

        clip_img_embeddings.append(xfI)
        clip_txt_embeddings.append(xfT)

        # Infer latent variables Z_new from new text features
        Z_txt_mean, Z_txt_variance = utils.infer_prob_embeddings(model_text, likelihood_text, cfg.MODEL.LATENT_DIM, xfT)
        Z_img_mean, Z_img_variance = utils.infer_prob_embeddings(model_image, likelihood_image, cfg.MODEL.LATENT_DIM, xfI)

        learned_img_embeddings_mean.append(Z_img_mean.detach())
        learned_txt_embeddings_mean.append(Z_txt_mean.detach())
        learned_img_embeddings_variance.append(Z_img_variance.detach())
        learned_txt_embeddings_variance.append(Z_txt_variance.detach())
    
r_dict['i_u'] = learned_img_embeddings_variance
r_dict['t_u'] = learned_txt_embeddings_variance

clip_img_embeddings = torch.cat(clip_img_embeddings).cpu()
clip_txt_embeddings = torch.cat(clip_txt_embeddings).cpu()
learned_img_embeddings_mean = torch.cat(learned_img_embeddings_mean).cpu()
learned_txt_embeddings_mean = torch.cat(learned_txt_embeddings_mean).cpu()
learned_img_embeddings_variance = torch.cat(learned_img_embeddings_variance).cpu()
learned_txt_embeddings_variance = torch.cat(learned_txt_embeddings_variance).cpu()

im_uncertainty=r_dict['i_u']
txt_uncertainty=r_dict['t_u']

pred_ranks_all = utils.get_pred_ranks(clip_img_embeddings, clip_txt_embeddings)

#ece_score = get_expected_calibration_error_retrieval(pred_ranks_all, q_classes_all=gt_labels, g_classes_all=gt_labels, embedding_variances=learned_txt_embeddings_variance)

recall_scores = utils.get_recall(pred_ranks_all, q_classes_all=gt_labels, g_classes_all=gt_labels)
#recall_scores = utils.get_recall_COCOFLICKR(pred_ranks_all, q_idx=q_idx)

print("Recall scores: ", recall_scores)

sort_v_idx, sort_t_idx = utils.sort_wrt_uncer(r_dict)

ret_bins_spacing = utils.create_uncer_bins_eq_spacing(sort_v_idx, n_bins=5)
ret_bins_samples = utils.create_uncer_bins_eq_samples(sort_v_idx, n_bins=5)


new_ece_score = utils.get_expected_calibration_error_retrieval(pred_ranks_all, q_classes_all=gt_labels, g_classes_all=gt_labels, embedding_variances=im_uncertainty, bins=ret_bins_spacing)

