library(tidyverse)

# specify parameters
n_networks <- 2
n_campaign_types <- 2
n_countries <- 10
n_video_orientations <- 3
n_app_categories <- 10
n_apps <- 50
n_videos <- 100

# generate names
networks <- paste0("network_", 1:n_networks)
campaign_types <- paste0("type_", 1:n_campaign_types)
countries <- paste0("country_", 1:n_countries)
video_orientations <- paste0("orientation_", 1:n_video_orientations)
app_categories <- paste0("category_", 1:n_app_categories)
apps <- paste0("app_", 1:n_apps)
videos <- paste0("video_", 1:n_videos)

# define number elements
n_rows <- n_videos * n_networks * n_campaign_types * n_countries

dimensions <- c("video", "orientation", "network", "type", "country", "app", "app_category")
metrics <- c("installs", "impressions", "ipm")
n_dimensions <- length(dimensions)
n_metrics <- length(metrics)
n_features <- 10
features <- paste0("feature_", 1:n_features)


binary_features_prob <- c(0.3, 0.6, 0.3, 0.8, 0.5)
continuous_features_alpha <- c(0.3, 0.6, 1, 0.8, 0.5)
continuous_features_beta <- c(0.2, 1.2, 0.8, 0.4, 0.5)

# create empty dataset
video_dataset <- data.frame(matrix(ncol = (n_dimensions + n_metrics + n_features),
                               nrow = n_rows))

# create names
names(video_dataset) <- c(dimensions, metrics, features)

# simulate dimensions
video_dataset$video <- rep(videos, each = n_rows / n_videos)
video_dataset$network <- rep(networks, n_rows / n_networks)
video_dataset$type <- rep(campaign_types, each = n_networks)
video_dataset$country <- rep(countries, each = (n_campaign_types * n_networks))
video_dataset$app <- rep(apps, each = (n_videos / n_apps * n_campaign_types * n_networks * n_countries))
video_dataset$app_category <- rep(app_categories, each = (n_videos / n_app_categories * n_campaign_types * n_networks * n_countries))
video_dataset <- video_dataset %>% 
  mutate(orientation = video_orientations[as.numeric(str_extract(video, "\\d.*$")) %% 3 + 1])

# simulate metrics
video_dataset$ipm <- rlnorm(nrow(video_dataset), meanlog = 1, sdlog = 0.5)

# simulate features
# binary features
video_dataset <- video_dataset %>% 
  mutate(feature_1 = rbinom(nrow(.), 1, binary_features_prob[1]),
         feature_2 = rbinom(nrow(.), 1, binary_features_prob[2]),
         feature_3 = rbinom(nrow(.), 1, binary_features_prob[3]),
         feature_4 = rbinom(nrow(.), 1, binary_features_prob[4]),
         feature_5 = rbinom(nrow(.), 1, binary_features_prob[5]))

# continuous features
video_dataset <- video_dataset %>% 
  mutate(feature_6 = round(rbeta(nrow(.), continuous_features_alpha[1], continuous_features_beta[1]), 4),
         feature_7 = round(rbeta(nrow(.), continuous_features_alpha[2], continuous_features_beta[2]), 4),
         feature_8 = round(rbeta(nrow(.), continuous_features_alpha[3], continuous_features_beta[3]), 4),
         feature_9 = round(rbeta(nrow(.), continuous_features_alpha[4], continuous_features_beta[4]), 4),
         feature_10 = round(rbeta(nrow(.), continuous_features_alpha[5], continuous_features_beta[5]), 4))
         
# network_2 has 20% lower ipm
video_dataset <- video_dataset %>% 
  mutate(ipm = if_else(network == "network_2", ipm * 0.9, ipm * 1.1))
# campaign_type_1 has 10% higher ipm
video_dataset <- video_dataset %>% 
  mutate(ipm = if_else(type == "type_1", ipm * 1.1, ipm * 0.9))

# app category adjustments
video_dataset <- video_dataset %>% 
  mutate(ipm = case_when(
    orientation == "orientation_1" ~ ipm * 1.1, 
    orientation == "orientation_2" ~ ipm * 0.8,
    TRUE ~ ipm
  ))

# country adjustments
video_dataset <- video_dataset %>% 
  mutate(ipm = case_when(
    country == "country_1" ~ ipm * 1.2, 
    country == "country_5" ~ ipm * 0.9,
    country == "country_9" ~ ipm * 0.5,
    TRUE ~ ipm
  ))

# app category adjustments
video_dataset <- video_dataset %>% 
  mutate(ipm = case_when(
    app_category == "category_2" ~ ipm * 2.5, 
    app_category == "category_4" ~ ipm * 1.5,
    app_category == "category_6" ~ ipm * 0.5,
    TRUE ~ ipm
  ))

# app adjustments
video_dataset <- video_dataset %>% 
  mutate(ipm = case_when(
    app == "app_10" ~ ipm * 1.1,
    app == "app_20" ~ ipm * 1.5,
    app == "app_30" ~ ipm * 1.7,
    app == "app_40" ~ ipm * 1.9,
    app == "app_50" ~ ipm * 1.5,
    app == "app_60" ~ ipm * 2.0,
    app == "app_70" ~ ipm * 0.9,
    app == "app_80" ~ ipm * 0.7,
    app == "app_90" ~ ipm * 0.6,
    app == "app_100" ~ ipm * 0.4,
    TRUE ~ ipm
  ))

# simulate slope importance
video_dataset_reg <- video_dataset %>% 
  mutate(ipm = ipm + ipm * feature_1 * 0.3,
         ipm = ipm + ipm * feature_3 * (-0.3),
         ipm = ipm + ipm * feature_6 * 0.2,
         ipm = ipm + ipm * feature_10 * (-0.4)
         )

# calculate difference from average value
video_dataset_reg <- video_dataset_reg %>% 
  group_by(app_category, app, country, network) %>% 
  mutate(avg_ipm = mean(ipm)) %>% 
  ungroup() %>% 
  mutate(ipm_diff = ipm - avg_ipm) %>% 
  select(ipm_diff, ipm, avg_ipm, everything())
