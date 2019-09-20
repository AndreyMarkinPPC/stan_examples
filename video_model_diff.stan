data {
  int<lower=1> N; // # of videos
  int<lower=1> K_features; // # of predictors from video features
  int<lower=1> N_apps; // # of apps
  int<lower=1> N_countries; // # of countries
  int<lower=1> N_types; // # of uac types
  int<lower=1> N_networks; // # of networks
  int<lower=1> N_orientations; // # of orientations
  
  int<lower=1, upper = N_apps> app_ids[N];
  int<lower=1, upper = N_countries> country_ids[N];
  int<lower=1, upper = N_types> type_ids[N];
  int<lower=1, upper = N_networks> network_ids[N];
  int<lower=1, upper = N_orientations> orientation_ids[N];
  
  
    // design matrices
  row_vector[K_features] X_features[N];           //fixed effects design matrix
  row_vector[K_features] Z_app[N];       //app random effects design matrix
  
  // response
  real<lower=0> y[N]; // vector of targets
}

parameters {
  
  vector[K_features] beta_features;             //fixed effects coefficients
  cholesky_factor_corr[K_features] L_app;  //cholesky factor of app random effects correlation matrix
  cholesky_factor_corr[K_features] L_country;  //cholesky factor of app random effects correlation matrix
  cholesky_factor_corr[K_features] L_type;  //cholesky factor of app random effects correlation matrix
  cholesky_factor_corr[K_features] L_network;  //cholesky factor of app random effects correlation matrix
  cholesky_factor_corr[K_features] L_orientation;  //cholesky factor of app random effects correlation matrix
  
  vector<lower=0>[K_features] sigma_app; //apps random effects standard defiation
  vector<lower=0>[K_features] sigma_country; //apps random effects standard defiation
  vector<lower=0>[K_features] sigma_type; //apps random effects standard defiation
  vector<lower=0>[K_features] sigma_network; //apps random effects standard defiation
  vector<lower=0>[K_features] sigma_orientation; //apps random effects standard defiation
  
  
  vector[K_features] z_u[N_apps];           //spherical app random effects
  vector[K_features] c_u[N_countries];           //spherical countries random effects
  vector[K_features] t_u[N_types];           //spherical types random effects
  vector[K_features] n_u[N_networks];           //spherical network random effects
  vector[K_features] s_u[N_orientations];           //spherical orientations random effects
  
  // ncp
  vector[N_apps] mu_raw;
  vector[N_countries] mu_raw_c;
  vector[N_types] mu_raw_t;
  vector[N_orientations] mu_raw_s;
  vector[N_networks] mu_raw_n;
  
  real<lower=0> sigma_mu;
  
  // variance
  real<lower=0> sigma;
  
}

transformed parameters {
  
  
  vector[K_features] app_ranef[N_apps];             //app random effects
  vector[K_features] country_ranef[N_countries];             //country random effects
  vector[K_features] type_ranef[N_types];             //type random effects
  vector[K_features] network_ranef[N_networks];             //network random effects
  vector[K_features] orientation_ranef[N_orientations];   //orientation random effects
 
 
  {
    matrix[K_features,K_features] Sigma_app;    //app random effects covariance matrix
    Sigma_app = diag_pre_multiply(sigma_app,L_app);
    
    for(j in 1:N_apps)
      app_ranef[j] = Sigma_app * z_u[j] + sigma_mu * mu_raw[j];
   }
   
  {
   matrix[K_features,K_features] Sigma_country;    //app random effects covariance matrix
    Sigma_country = diag_pre_multiply(sigma_country,L_country);
    
   for(j in 1:N_countries)
      country_ranef[j] = Sigma_country * c_u[j] + sigma_country * mu_raw_c[j];
   }
   
  {
   matrix[K_features,K_features] Sigma_type;    //app random effects covariance matrix
    Sigma_type = diag_pre_multiply(sigma_type,L_type);
   for(j in 1:N_types)
      type_ranef[j] = Sigma_type * t_u[j] + sigma_type * mu_raw_t[j];
   }
   
  {
    matrix[K_features,K_features] Sigma_network;    //app random effects covariance matrix
    Sigma_network = diag_pre_multiply(sigma_network,L_network);
   for(j in 1:N_networks)
      network_ranef[j] = Sigma_network * n_u[j] + sigma_network * mu_raw_n[j];
   }
   
  {
    matrix[K_features,K_features] Sigma_orientation;    //app random effects covariance matrix
    Sigma_orientation = diag_pre_multiply(sigma_orientation,L_orientation);
   for(j in 1:N_orientations)
      orientation_ranef[j] = Sigma_orientation * s_u[j] + sigma_orientation * mu_raw_s[j];
   }
}

model {
  
  L_app ~ lkj_corr_cholesky(2.0);
  L_country ~ lkj_corr_cholesky(2.0);
  L_type ~ lkj_corr_cholesky(2.0);
  L_network ~ lkj_corr_cholesky(2.0);
  L_orientation ~ lkj_corr_cholesky(2.0);
  
  beta_features ~ normal(0, 1);
  mu_raw ~ normal(0, 1);
  sigma_mu ~ normal(0, 1);
  
  for (j in 1:N_apps)
    z_u[j] ~ normal(0,1);
    
  for (j in 1:N_countries)
    c_u[j] ~ normal(0,1);
    
  for (j in 1:N_types)
    t_u[j] ~ normal(0,1);
    
  for (j in 1:N_networks)
    n_u[j] ~ normal(0,1);
  
  for (j in 1:N_orientations)
    s_u[j] ~ normal(0,1);
  
  // Likelihood
  for (i in 1:N)  
    y[i] ~ normal(X_features[i] * beta_features + 
    Z_app[i] * app_ranef[app_ids[i]] + 
    Z_app[i] * country_ranef[country_ids[i]] + 
    Z_app[i] * type_ranef[type_ids[i]] +
    Z_app[i] * network_ranef[network_ids[i]] +
    Z_app[i] * orientation_ranef[orientation_ids[i]], sigma);
  
}


generated quantities {
  vector[N] y_rep;
  //vector[N] log_lik;
  
  for(i in 1:N) {
    y_rep[i] = normal_rng(X_features[i] * beta_features + 
    Z_app[i] * app_ranef[app_ids[i]] + 
    Z_app[i] * country_ranef[country_ids[i]] + 
    Z_app[i] * type_ranef[type_ids[i]] +
    Z_app[i] * network_ranef[network_ids[i]] +
    Z_app[i] * orientation_ranef[orientation_ids[i]], sigma);
    
    //log_lik[i] = lognormal_lpdf(y[i] | X_features[i] * beta_features + Z_app[i] * app_ranef[app_ids[i]], sigma);
  }
  
}
