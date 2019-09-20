library(rstan)
library(dplyr)
video_dataset_reg <- readRDS("~/Downloads/video_dataset_reg.RDS")
video_dataset_reg <- video_dataset_reg %>% 
  mutate(app_id = as.integer(as.factor(app)),
         app_category_id = as.integer(as.factor(app_category)),
         orientation_id = as.integer(as.factor(orientation)),
         network_id = as.integer(as.factor(network)),
         type_id = as.integer(as.factor(type)),
         country_id = as.integer(as.factor(country))
         ) 
data_stan_list <- list(
  N = nrow(video_dataset_reg), 
  N_apps = n_apps, 
  N_types = n_campaign_types, 
  N_countries = n_countries, 
  N_networks = n_networks, 
  N_orientations = n_video_orientations,
  app_ids = video_dataset_reg$app_id,
  country_ids = video_dataset_reg$country_id, 
  type_ids = video_dataset_reg$type_id,
  network_ids = video_dataset_reg$network_id, 
  orientation_ids = video_dataset_reg$orientation_id,
  K_features = n_features + 1,
  X_features = cbind(1, video_dataset_reg[, 11:20]), 
  Z_app = cbind(1, video_dataset_reg[, 11:20]), 
  y = video_dataset_reg$ipm_diff
)

model <- stan_model("~/Downloads/video_model_diff.stan")
model_samples_vb <- vb(model, data = data_stan_list, 
                           pars = c("beta_features", 
                                    "app_ranef", 
                                    "country_ranef", 
                                    "type_ranef", 
                                    "network_ranef",
                                    "orientation_ranef",
                                    "y_rep"))
                                    
# explore parameters
print(model_samples_vb, pars = "beta_features")
print(model_samples_vb, pars = "app_ranef")
                                    
# visualize
mcmc_areas(x = as.matrix(model_samples_vb), 
           regex_pars = "beta_features", 
           prob = 0.8) + 
  ggtitle("Posterior distribution of beta_features", "with medians and 80% CI")

# ppc check
y_rep <- as.matrix(model_samples_vb, pars = "y_rep")
ppc_dens_overlay(y = data_stan_list$y, 
                 yrep = y_rep[1:200, ]) + 
  xlim(0, 50)
