from catboost import CatBoostRegressor
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import optuna
import pandas as pd
import skgstat as skg
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, train_test_split
from tqdm.auto import tqdm
from xgboost import XGBRegressor

from PyGRF import PyGRFBuilder, search_bw_lw_ISA
from supertree import SuperTree


def determine_optimal_block_size(df, target_col, max_sample=5000, harvester_width=10):
    print("--- Calculating Spatial Autocorrelation Range ---")
    
    coords = np.column_stack((df.geometry.centroid.x, df.geometry.centroid.y))
    y_vals = df[target_col].values
    
    if len(df) > max_sample:
        np.random.seed(42)
        idx = np.random.choice(len(df), max_sample, replace=False)
        coords = coords[idx]
        y_vals = y_vals[idx]
    
    variogram = skg.Variogram(coords, y_vals, model='spherical', maxlag='median')
    raw_range = variogram.parameters[0]
    
    t = raw_range % harvester_width
    if t < (harvester_width / 2):
        optimal_block_size = int(np.floor(raw_range / harvester_width) * harvester_width)
    else:
        optimal_block_size = int(np.ceil(raw_range / harvester_width) * harvester_width)

    print(f"Raw Range: {raw_range:.1f}m | Snapped Block Size: {optimal_block_size}m")
    
    fig, ax = plt.subplots(figsize=(6,4))
    variogram.plot(axes=ax, hist=False)
    ax.set_title("Yield Semivariogram (Spatial Autocorrelation)")
    ax.axvline(raw_range, color='red', linestyle='--', label=f'Calculated Range ({raw_range:.1f}m)')
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    return optimal_block_size


def spatial_checkerboard_split(df, pred_params, target_col, block_size, test_size=0.2, random_state=42):
    print(f"\n--- Generating Spatial Block Split (Block Size: {block_size}m) ---")
    
    clean_df = df.copy()
    
    minx, miny, _, _ = clean_df.total_bounds
    clean_df['block_x'] = ((clean_df.geometry.centroid.x - minx) // block_size).astype(int)
    clean_df['block_y'] = ((clean_df.geometry.centroid.y - miny) // block_size).astype(int)
    clean_df['block_id'] = clean_df['block_x'].astype(str) + "_" + clean_df['block_y'].astype(str)
    
    unique_blocks = clean_df['block_id'].unique()
    train_blocks, test_blocks = train_test_split(unique_blocks, test_size=test_size, random_state=random_state)
    
    train_df = clean_df[clean_df['block_id'].isin(train_blocks)].copy()
    test_df = clean_df[clean_df['block_id'].isin(test_blocks)].copy()
    
    X_train = train_df[pred_params].copy()
    y_train = train_df[target_col].copy()
    X_test = test_df[pred_params].copy()
    y_test = test_df[target_col].copy()
    
    print(f"Total Blocks: {len(unique_blocks)}")
    print(f"  -> Training: {len(train_blocks)} blocks ({len(X_train):,} polygons)")
    print(f"  -> Testing:  {len(test_blocks)} blocks ({len(X_test):,} polygons)")
    
    return X_train, X_test, y_train, y_test, train_df, test_df


def plot_checkerboard_split(train_df, test_df):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title("Training and testing blocks")
    
    train_df.plot(ax=ax, color='lightgray', edgecolor='none')
    test_df.plot(ax=ax, color='crimson', edgecolor='none')
    
    train_patch = mpatches.Patch(color='lightgray', label=f'Training Data')
    test_patch = mpatches.Patch(color='crimson', label=f'Testing Data')
    ax.legend(handles=[train_patch, test_patch], loc='upper right')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.axis('equal')
    ax.set_xlabel('Easting')
    ax.set_ylabel('Northing')
    plt.axis('equal') # Prevents distortion of the field map
    plt.tight_layout()
    plt.show()


def rf_predict_spatial(input_df, pred_params, target_col, block_size=50, model_name = 'rf_spatial', model_params = None, cv = True):
    """   
    Performs Random Forest regression using Spatial Block Cross-Validation 
    with a progress bar to track execution.
    """
    
    with tqdm(total=5, desc="Initializing Spatial RF", unit="step") as pbar:
        
        # --- STEP 1: Prep Data & Blocks ---
        
        clean_df = input_df.copy() # .dropna(subset=pred_params + [target_col]).copy()
        X, y = clean_df[pred_params].copy(), clean_df[target_col].copy()
        
        if not cv:
            pbar.write("Cross-validation is disabled. Proceeding with a single train-test split.")
            if model_params:
                print("Training model using provided hyperparameters")
                rfr_model = RandomForestRegressor(**model_params, random_state=42, n_jobs=-1, oob_score=True, verbose = 0) 
            else:
                rfr_model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1, oob_score=True, verbose = 0) 

            rfr_model.fit(X, y)
                            
            return rfr_model, None
        
        pbar.set_description("Step 1/5: Preparing Data & Spatial Blocks")

        
        minx, miny, _, _ = clean_df.total_bounds
        clean_df['block_x'] = ((clean_df.geometry.centroid.x - minx) // block_size).astype(int)
        clean_df['block_y'] = ((clean_df.geometry.centroid.y - miny) // block_size).astype(int)
        groups = clean_df['block_x'].astype(str) + "_" + clean_df['block_y'].astype(str)
        
        pbar.write(f"Created {groups.nunique()} unique spatial blocks for cross-validation.")
        pbar.update(1)

        # --- STEP 2: Train Folds ---
        pbar.set_description("Step 2/5: Training Across Spatial Folds")
        
        gkf = GroupKFold(n_splits=5)
        if model_params:
            print("Training model using provided hyperparameters")
            rfr_model = RandomForestRegressor(**model_params, random_state=42, n_jobs=-1, oob_score=True, verbose = 0) 
        else:
            rfr_model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1, oob_score=True, verbose = 0) 

        oof_predictions = np.zeros(len(y))
        
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            rfr_model.fit(X_train, y_train)
            preds = rfr_model.predict(X_test)
            oof_predictions[test_idx] = preds
            
            # Use pbar.write instead of print to avoid breaking the progress bar visually
            pbar.write(f"  -> Fold {fold+1} Spatial R²: {r2_score(y_test, preds):.3f}")
            
        pbar.update(1)

        # --- STEP 3: Final Metrics ---
        pbar.set_description("Step 3/5: Calculating Metrics")
        final_r2 = r2_score(y, oof_predictions)
        final_rmse = root_mean_squared_error(y, oof_predictions)
        
        pbar.write("\n--- Final Spatial Out-of-Fold Results ---")
        pbar.write(f"R²: {final_r2:.3f}")
        pbar.write(f"RMSE: {final_rmse:.3f}")
        pbar.write("-----------------------------------------")
        pbar.update(1)

        # --- STEP 4: Final Model Fit ---
        pbar.set_description("Step 4/5: Training Final Full Model")
        rfr_model.fit(X, y)
        pbar.update(1)
        
        # --- STEP 5: Plotting ---
        pbar.set_description("Step 5/5: Calculating Permutation Importance & Plotting")
        pbar.write("Calculating permutation importances (this may take a moment)...")
        perm_results = permutation_importance(rfr_model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
        
        # Extract the mean importances (how much R2 dropped when shuffled)
        importances = perm_results.importances_mean
        features = X.columns
        
        results_dict  = {
            'model_name': model_name,
            'model': rfr_model,
            'y_true': y.values,
            'y_pred': oof_predictions,
            'importances': perm_results.importances_mean,
            'features': X.columns
        }
        
        
        # importances, features = rfr_model.feature_importances_, X.columns
        
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        
        axes[0].scatter(y, oof_predictions, alpha=0.3, s=5)
        axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        axes[0].set_xlabel('Actual Yield')
        axes[0].set_ylabel('Predicted Yield (Out-of-Fold)')
        axes[0].set_title('Actual vs Predicted yield')
                
        sorted_idx = np.argsort(importances)[::-1]
        if len(sorted_idx) >= 11:
            sorted_idx = sorted_idx[0:10]
            r = 10
        else:
            r = len(sorted_idx)
            
        axes[1].bar(range(r), importances[sorted_idx])
        axes[1].set_xticks(range(r))
        axes[1].set_xticklabels(features[sorted_idx], rotation=45, ha='right')
        axes[1].set_title("Permutation Feature Importances")
        plt.tight_layout()
        plt.show()


        fig, axes = plt.subplots(1, 1, figsize=(6,5))       
        axes.scatter(y, oof_predictions, alpha=0.3, s=5)
        axes.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        axes.set_xlabel('Actual Yield')
        axes.set_ylabel('Predicted Yield (Out-of-Fold)')
        axes.set_title('Actual vs Predicted')
        plt.tight_layout()
        plt.show()


        fig, axes = plt.subplots(1, 1, figsize=(6,5))   
        axes.bar(range(r), importances[sorted_idx])
        axes.set_xticks(range(r))
        axes.set_xticklabels(features[sorted_idx], rotation=45, ha='right')
        axes.set_title("Permutation Feature Importances")
        plt.tight_layout()
        plt.show()
        
        pbar.update(1)
        pbar.set_description("Complete")
        
    return rfr_model, results_dict


def xgb_predict_spatial(input_df, pred_params, target_col, block_size=50, model_name = 'xgb_spatial', model_params = None, cv = True):

    
    with tqdm(total=5, desc="Initializing Spatial RF", unit="step") as pbar:
        
        # --- STEP 1: Prep Data & Blocks ---
        pbar.set_description("Step 1/5: Preparing Data & Spatial Blocks")
        
        clean_df = input_df.copy() # .dropna(subset=pred_params + [target_col]).copy()
        X, y = clean_df[pred_params].copy(), clean_df[target_col].copy()
        
        if cv:
            minx, miny, _, _ = clean_df.total_bounds
            clean_df['block_x'] = ((clean_df.geometry.centroid.x - minx) // block_size).astype(int)
            clean_df['block_y'] = ((clean_df.geometry.centroid.y - miny) // block_size).astype(int)
            groups = clean_df['block_x'].astype(str) + "_" + clean_df['block_y'].astype(str)
            
            pbar.write(f"Created {groups.nunique()} unique spatial blocks for cross-validation.")
            pbar.update(1)

            # --- STEP 2: Train Folds ---
            pbar.set_description("Step 2/5: Training Across Spatial Folds")
        
            gkf = GroupKFold(n_splits=5)
            if model_params:
                print("Training model using provided hyperparameters")
                xgb_model = XGBRegressor(**model_params, random_state=42, n_jobs=-1) 
            else:
                xgb_model = XGBRegressor(n_estimators=150, random_state=42, n_jobs=-1) 

            oof_predictions = np.zeros(len(y))
            
            for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                xgb_model.fit(X_train, y_train)
                preds = xgb_model.predict(X_test)
                oof_predictions[test_idx] = preds
                
                # Use pbar.write instead of print to avoid breaking the progress bar visually
                pbar.write(f"  -> Fold {fold+1} Spatial R²: {r2_score(y_test, preds):.3f}")        
        else:
            if model_params:
                print("Training model using provided hyperparameters")
                xgb_model = XGBRegressor(**model_params, random_state=42, n_jobs=-1) 
            else:
                xgb_model = XGBRegressor(n_estimators=150, random_state=42, n_jobs=-1)

            xgb_model.fit(X, y)
                            
            return xgb_model, None
                
        pbar.update(1)

        # --- STEP 3: Final Metrics ---
        pbar.set_description("Step 3/5: Calculating Metrics")
        final_r2 = r2_score(y, oof_predictions)
        final_rmse = root_mean_squared_error(y, oof_predictions)
        
        pbar.write("\n--- Final Spatial Out-of-Fold Results ---")
        pbar.write(f"R²: {final_r2:.3f}")
        pbar.write(f"RMSE: {final_rmse:.3f}")
        pbar.write("-----------------------------------------")
        pbar.update(1)

        # --- STEP 4: Final Model Fit ---
        pbar.set_description("Step 4/5: Training Final Full Model")
        xgb_model.fit(X, y)
        pbar.update(1)
        
        # --- STEP 5: Plotting ---        
        pbar.set_description("Step 5/5: Calculating Permutation Importance & Plotting")
        pbar.write("Calculating permutation importances (this may take a moment)...")
        perm_results = permutation_importance(xgb_model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
        
        # Extract the mean importances (how much R2 dropped when shuffled)
        importances = perm_results.importances_mean
        features = X.columns
        
        results_dict  = {
            'model_name': model_name,
            'model': xgb_model,
            'y_true': y,
            'y_pred': oof_predictions,
            'importances': importances,
            'features': features
        }
        
        
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        
        axes[0].scatter(y, oof_predictions, alpha=0.3, s=5)
        axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        axes[0].set_xlabel('Actual Yield')
        axes[0].set_ylabel('Predicted Yield (Out-of-Fold)')
        axes[0].set_title('Actual vs Predicted yield')
        
        # importances, features = xgb_model.feature_importances_, X.columns
        sorted_idx = np.argsort(importances)[::-1]
        if len(sorted_idx) >= 11:
            sorted_idx = sorted_idx[0:10]
            r = 10
        else:
            r = len(sorted_idx)
            
        axes[1].bar(range(r), importances[sorted_idx])
        axes[1].set_xticks(range(r))
        axes[1].set_xticklabels(features[sorted_idx], rotation=45, ha='right')
        axes[1].set_title("Feature Importances")
        plt.tight_layout()
        plt.show()
        
        fig, axes = plt.subplots(1, 1, figsize=(6,5))       
        axes.scatter(y, oof_predictions, alpha=0.3, s=5)
        axes.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        axes.set_xlabel('Actual Yield')
        axes.set_ylabel('Predicted Yield (Out-of-Fold)')
        axes.set_title('Actual vs Predicted yield')
        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(1, 1, figsize=(6,5))   
        axes.bar(range(r), importances[sorted_idx])
        axes.set_xticks(range(r))
        axes.set_xticklabels(features[sorted_idx], rotation=45, ha='right')
        axes.set_title("Permutation Feature Importances")
        plt.tight_layout()
        plt.show()
        
        pbar.update(1)
        pbar.set_description("Complete")
        
    return xgb_model, results_dict


def tune_spatial_hyperparameters(input_df, pred_params, target_col, base_model, param_dist, n_iter=10, block_size=50, cat_features=None):
    """
    Optionally tunes hyperparameters for a single model using Spatial Cross-Validation.
    Supports both standard scikit-learn models and CatBoost.
    
    Args:
        input_df (GeoDataFrame): The master grid.
        pred_params (list): List of predictor columns.
        target_col (str): Target column to predict.
        base_model: The scikit-learn compatible model to tune (e.g., CatBoostRegressor()).
        param_dist (dict): Dictionary with parameters names as keys and distributions to try.
        n_iter (int): Number of random parameter settings to sample.
        block_size (int): Size in meters of the hold-out blocks.
        cat_features (list, optional): List of categorical column names for CatBoost.
        
    Returns:
        best_model: The model instantiated with the best found parameters.
        best_params (dict): The dictionary of the best parameters.
    """
    print(f"Starting Spatial Hyperparameter Tuning ({n_iter} iterations)...")
    
    # 1. Prepare Data & Spatial Blocks
    clean_df = input_df.copy() # .dropna(subset=pred_params + [target_col]).copy()
    X, y = clean_df[pred_params].copy(), clean_df[target_col].copy()
    
    minx, miny, _, _ = clean_df.total_bounds
    clean_df['block_x'] = ((clean_df.geometry.centroid.x - minx) // block_size).astype(int)
    clean_df['block_y'] = ((clean_df.geometry.centroid.y - miny) // block_size).astype(int)
    groups = clean_df['block_x'].astype(str) + "_" + clean_df['block_y'].astype(str)
    
    # 2. Setup Spatial CV
    gkf = GroupKFold(n_splits=5)
    # Generate the splits as a list so RandomizedSearchCV respects the spatial groups
    cv_splits = list(gkf.split(X, y, groups=groups))
    
    # 3. Setup Randomized Search
    # We use negative RMSE as the scoring metric because scikit-learn tries to maximize the score
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        #scoring = 'r2',
        scoring='neg_root_mean_squared_error',
        cv=cv_splits,
        verbose=2,       # Outputs progress to the console
        random_state=42,
        n_jobs=-1        # Uses all available CPU cores
    )
    
    # 4. Execute Search
    # Package any necessary fit parameters (like cat_features for CatBoost)
    fit_params = {}
    if cat_features is not None:
        fit_params['cat_features'] = cat_features
    
    search.fit(X, y)
    
    print("\n--- Tuning Complete ---")
    print(f"Best Spatial RMSE: {np.abs(search.best_score_):.3f}")
    print("Best Parameters:", search.best_params_)
    print("-----------------------")
    
    # Return the optimized model ready for your final comparison
    return search.best_estimator_, search.best_params_


def tune_spatial_optuna(input_df, pred_params, target_col, model_type='xgb', n_trials=30, block_size=50):
    """
    Uses Optuna (Bayesian Optimization) to find the best hyperparameters 
    while strictly enforcing Spatial Block Cross-Validation.
    
    Args:
        input_df (GeoDataFrame): The master grid.
        pred_params (list): List of predictor columns.
        target_col (str): Target column to predict.
        model_type (str): 'xgb' for XGBoost or 'rf' for Random Forest.
        n_trials (int): Number of optimization trials to run.
        block_size (int): Size in meters of the hold-out blocks.
        
    Returns:
        best_model: The fully trained model with the best parameters.
        study: The Optuna study object (useful for plotting optimization history).
    """
    print(f"Starting Optuna Optimization for {model_type.upper()} ({n_trials} trials)...")
    
    # 1. Prepare Data & Spatial Blocks
    clean_df = input_df.dropna(subset=pred_params + [target_col]).copy()
    X, y = clean_df[pred_params], clean_df[target_col]
    
    minx, miny, _, _ = clean_df.total_bounds
    clean_df['block_x'] = ((clean_df.geometry.centroid.x - minx) // block_size).astype(int)
    clean_df['block_y'] = ((clean_df.geometry.centroid.y - miny) // block_size).astype(int)
    groups = clean_df['block_x'].astype(str) + "_" + clean_df['block_y'].astype(str)

    # 2. Define the Objective Function for Optuna
    def objective(trial):
        # Dynamically suggest parameters based on the chosen model
        if model_type == 'rf':
            params = {
                # 'n_estimators': trial.suggest_int('n_estimators', 50, 650, step=50),
                # 'max_depth': trial.suggest_categorical('max_depth', [None, 5, 10, 20, 30, 40,50]),
                # 'min_samples_split': trial.suggest_categorical('min_samples_split', [2,3,4, 5, 10, 15, 20]),
                # 'max_features': trial.suggest_categorical('max_features', ["sqrt", "log2", 0.2, 0.5, 0.8, None]),
                # 'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [1,2,4,8,16]),
                # 'random_state': 42,
                # 'n_jobs': -1

                # Number of trees (more does not overfit, but bounds save time)
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                # Regularisation: Restrict tree size
                'max_depth': trial.suggest_int('max_depth', 3, 50),
                # Regularisation: Require more samples to split a node
                'min_samples_split': trial.suggest_int('min_samples_split', 4, 30),
                # Regularisation: Require more samples to form a leaf
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 15),
                # Regularisation: Force feature diversity per split
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.33, 0.5, 0.75]),                
                'random_state': 42,
                'n_jobs': -1
            }
            model = RandomForestRegressor(**params)
            
        elif model_type == 'xgb':
            params = {
                # 'n_estimators': trial.suggest_int('n_estimators', 50, 650, step=50),
                # 'learning_rate': trial.suggest_categorical('learning_rate', [0.005, 0.01, 0.05, 0.1, 0.2]),
                # 'max_depth': trial.suggest_categorical('max_depth', [3, 5, 7, 9, 11, 13, 15]),
                # 'subsample': trial.suggest_categorical('subsample', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                # 'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, None]),
                
                # Restrict estimators to prevent sequential overfitting
                'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
                # Lower learning rate requires more robust patterns
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
                # Regularisation: Restrict tree depth severely
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                # Regularisation: Hide a fraction of rows from each tree
                'subsample': trial.suggest_float('subsample', 0.5, 0.9),
                # Regularisation: Hide a fraction of features from each tree
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
                # # Minimum loss reduction required to make a further partition
                # 'gamma': trial.suggest_float('gamma', 0.1, 10.0, log=True),
                # # L1 regularisation on leaf weights (encourages sparsity)
                # 'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 10.0, log=True),
                # # L2 regularisation on leaf weights (smooths extreme weights)
                # 'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0, log=True),
                
                'random_state': 42,
                'n_jobs': -1
            }
            model = XGBRegressor(**params)
        else:
            raise ValueError("model_type must be 'rf' or 'xgb'")

        # Spatial Cross-Validation Loop
        gkf = GroupKFold(n_splits=5)
        oof_preds = np.zeros(len(y))
        
        for train_idx, test_idx in gkf.split(X, y, groups=groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            oof_preds[test_idx] = model.predict(X_test)
            
        # The metric we want Optuna to optimise
        #return r2_score(y, oof_preds)  # We want to maximize R², so we return it directly for maximisation
        return root_mean_squared_error(y, oof_preds) # We want to minimize RMSE, so we return it directly for minimisation

    # 3. Create and Run the Optuna Study
    # We suppress Optuna's default noisy logging and use a progress bar instead
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize') # minimise for rmse
    #study = optuna.create_study(direction='maximize') # maximise for r2
    
    # Run the optimization with a built-in progress bar
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print("\n--- Optuna Tuning Complete ---")
    print(f"Best Spatial RMSE: {study.best_value:.3f}")
    #print(f"Best Spatial R²: {study.best_value:.3f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("------------------------------")
    
    # 4. Train the Final Best Model on ALL Data
    if model_type == 'rf':
        best_model = RandomForestRegressor(**study.best_params, random_state=42, n_jobs=-1)
    else:
        best_model = XGBRegressor(**study.best_params, random_state=42, n_jobs=-1)
        
    best_model.fit(X, y)

    return best_model, study


