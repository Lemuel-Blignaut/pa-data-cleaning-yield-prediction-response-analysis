# Precision Agriculture Spatial Data Workflow

This repository contains an end-to-end precision agriculture data cleaning crop yield prediction and response analysis workflow. The workflow handles everything from raw yield map ingestion and sensor delay correction to spatial block cross-validation and model interpretability.

## Dependencies

To run this pipeline, you will need the following key Python libraries installed:

* **Geospatial & Data Processing:** `geopandas`, `pandas`, `numpy`, `shapely`, `scipy`
* **Machine Learning:** `scikit-learn`, `xgboost`, `pyGRF`
* **Spatial Analysis & Geostatistics:** `libpysal`, `skgstat` (SciKit-GStat)
* **Optimization:** `optuna`
* **Interpretability:** `shap`
* **Utilities:** `rapidfuzz`, `matplotlib`, `seaborn`, `tqdm`, `joblib`

---

## Project Structure & Modules

The workflow is divided into modular Python files, each handling a specific stage of the data pipeline:

### 1. `data_imports.py` - I/O and Standardization

Handles loading raw shapefiles and standardizing column names using fuzzy matching.

* **`files_to_dict(dir_path)`**: Recursively searches a directory for `.shp` files and loads them into a dictionary of GeoDataFrames.
* **`cast_to_crs(input_dict, crs)`**: Ensures all GeoDataFrames in a dictionary are projected to a uniform Coordinate Reference System (CRS).
* **`load_schema(json_path)`**: Loads the JSON dictionary used for standardizing variable names.
* **`rename_columns_fuzzy(df, schema_dict, threshold, exclude_cols)`**: Uses RapidFuzz to match raw column names to a standardized schema (defined in `schema_mapping.json`), ensuring consistent variable names across different datasets.

### 2. `preprocessing.py` - Spatial Filtering & Cleaning

Cleans the data by removing areas outside trial boundaries and filtering statistical/spatial anomalies.

* **`trial_boundaries(df)`**: Extracts the main trial boundaries by isolating the largest contiguous polygon.
* **`remove_boundary_data(input_df, trial_df, buffer_zone)`**: Clips data points falling outside the defined trial boundary.
* **`filter_parameter(input_df, seed_colname, parameter_colname, method)`**: Removes statistical outliers based on percentiles, hard bounds, or standard deviations.
* **`remove_local_outliers(gdf, value_col, crit, s, sd_threshold)`**: Detects and removes spatial anomalies (e.g., yield spikes) by comparing a point to its immediate spatial neighbors using a KDTree.
* **`remove_spatial_outliers(gdf, dist, n)`**: Iteratively drops points that lack sufficient spatial density (e.g., isolated GPS points).
* **`text_encoder(cols, df)`**: Applies One-Hot Encoding to categorical variables.

### 3. `delay.py` - Sensor Lag Correction

Corrects spatial offsets caused by combine harvester sensor delays (e.g., the time it takes grain to travel from the header to the mass flow sensor).

* **`manual_lag_correction(...)`**: Temporally shifts yield data by a static, manually defined time lag.
* **`run_pcdi_analysis(...)`**: Executes a single run of the Phase Correlation Delay Identification (PCDI) method using Fast Fourier Transforms (FFT) to find the optimal delay that aligns adjacent harvest passes.
* **`run_robust_pcdi(...)`**: Runs PCDI multiple times (adding Gaussian White Noise to missing pixels) to robustly determine the ideal sensor delay time and plots before/after spatial maps.

### 4. `terrain.py` - Topographic Derivatives

Extracts terrain features from 3D spatial points.

* **`terrain(x, y, z, crit, s)`**: Uses KDTree queries and parallel processing to fit localized 3D planes to point neighborhoods, calculating the **slope** and **aspect** (in degrees) for every spatial point.

### 5. `aggregation.py` - Spatial Gridding & Imputation

Standardizes variable-density point data into uniform spatial grids for ML modeling.

* **`calculate_dominant_heading(points_df)`**: Determines the primary driving angle of the harvester to align spatial grids.
* **`create_spatial_grid(input_df, cell_size, angle)`**: Generates a regular square fishnet grid, optionally rotated to match the dominant field heading.
* **`aggregate_points_to_grid(...)`**: Aggregates point data (e.g., GPS yield points) into grid cells, correctly handling both extensive (mass) and intensive (rate) variables.
* **`assign_polygon_to_grid(...)`**: Overlays management polygons (e.g., planting zones) onto the grid using area-weighted proportions.
* **`align_to_grid(master_grid, gdf, col, method)`**: Snaps source geometries to a master grid using Nearest Neighbor or Inverse Distance Weighting (IDW).
* **`impute_spatial_mean(...)`**: Fills `NaN` values in the grid using the average of their spatial neighbors (Queen, Rook, or KNN weights via `libpysal`).

### 6. `modelling.py` - Machine Learning & Spatial CV

Trains machine learning models while strictly accounting for spatial autocorrelation to prevent overfitting.

* **`determine_optimal_block_size(df, target_col, harvester_width)`**: Calculates the empirical semivariogram range to determine the mathematically optimal spatial block size for Cross-Validation.
* **`spatial_checkerboard_split(...)`**: Divides the field into a checkerboard pattern for spatial train/test splits.
* **`rf_predict_spatial(...)` / `xgb_predict_spatial(...)`**: Trains Random Forest or XGBoost models using Spatial Block GroupKFold Cross-Validation, preventing data leakage between adjacent points.
* **`compare_spatial_models(...)`**: Benchmarks multiple models against each other over the spatial folds.
* **`tune_spatial_optuna(...)`**: Performs Bayesian Hyperparameter Optimization using Optuna, utilizing spatial RMSE as the minimization objective to find the most robust parameters.
* **`train_final_model(...)`**: Trains the final production model on 100% of the dataset using the optimized hyperparameters.

### 7. `interpretation.py` - Model Diagnostics & XAI

Unpacks the "black box" models to understand what drives crop yield.

* **`run_full_diagnostics(...)`**: Generates a comprehensive suite of diagnostic plots including Residuals, Parity, out-of-fold RMSE/R2, Permutation Importance, and Gini Importance.
* **`shap_values(...)`**: Implements SHAP (SHapley Additive exPlanations) for global feature importance (Summary/Beeswarm plots) and local pixel-level explanations (Waterfall plots).
* **`plot_three_pdp_ice(...)` / `two_dim_pdp_plots(...)`**: Generates 1D and 2D Partial Dependence Plots (PDP) and Individual Conditional Expectation (ICE) curves to visualize the modeled relationships between inputs (e.g., fertilizer, seeding rate) and yield.
* **`plot_comparative_residuals(...)`**: Generates residual plots for the two models that are being compared.
* **`plot_comparative_parity(...)`**: Generates parity plots for the two models that are being compared.
* **`plot_comparative_importances(...)`**: Generates feautre importance plots for the two models that are being compared.

### 8. `spatial_random_forest_implementation.py` - GRF implementation and analysis

Implements the GRF model. Data split, training, testing, XAI.

* **`prep_dataset(...)`**: Adds coordinate columns to dataset, does train-test split, assigns feature columns.
* **`spatial_parameters(...)`**: Calculates the bandwidth and local weight needed to train the GRF model
* **`train_grf_model(...)`**: Trains model, calculates global and local feature importances and returns all of the above as well as combined and global predictions.
* **`prep_results(...)`**: Prepares residual and prediction dataframes, plots global feature importances
* **`plot_local_r2_error(...)`**: Calculates and plots R2 error on a map of the field.
* **`plot_residuals(...)`**: Plots residuals on a map of the field.
* **`plot_local_feat_imp(...)`**: Plots the local feature importance of one or more features on maps of the field. Recommended to plot no more than two features at a time.
* **`plot_actual_pred_res(...)`**: Plots the recorded and predicted yield as well as the residuals on a 1x3 plot.
* **`plot_parity_residuals(...)`**: Shows the parity plot as well as residual plot, this time not as a map but only as a graph.

## `schema_mapping.json`

This JSON file contains a dictionary of aliases for various precision ag metrics (e.g., linking "Yld_Mass_D", "DRYMATTER", and "VRYIELDMAS" to the standard target `yield_mass_dry`). It is utilized by `data_imports.py` to ensure seamless processing regardless of the machinery or monitor brand used to collect the data.
