from IPython import display
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, cross_val_score


def plot_three_pdp_ice(model, X_train, target_features):
    """
    Plots combination ICE and PDP plots for three specified features.
    
    Parameters:
    - model: Fitted scikit-learn model.
    - X_train: The training dataset (DataFrame) to evaluate the learned relationships.
    - target_features: A list of exactly three feature names (strings).
    """
    if len(target_features) != 3:
        raise ValueError("Please provide exactly three feature names in the target_features list.")
        
    fig, ax = plt.subplots(1, 3, figsize=(11, 3), constrained_layout=True, sharey=True)
    
    # --- Define Plot Styling ---
    ice_style = {
        "linewidth": 0.5,       # Make them very thin
        "linestyle": "-",       # Solid line 
        "color": "grey",   # Push them to the background
        "alpha": 0.5            # Make them semi-transparent
    }
    
    pdp_style = {
        "linewidth": 3.0,       # Make it thick and prominent
        "linestyle": "--",      # Dashed line
        "color": "crimson"      # Bright contrasting colour
    }
    
    # kind="both" forces scikit-learn to plot both ICE lines and the average PDP line
    display = PartialDependenceDisplay.from_estimator(
        estimator=model,
        X=X_train,
        features=target_features,
        kind="both",
        ax=ax,
        n_jobs=-1,
        ice_lines_kw=ice_style,
        pd_line_kw=pdp_style
    )
    ax[0].set_ylabel("Predicted Yield (t/ha)")
    ax[1].set_ylabel("Predicted Yield (t/ha)")
    ax[2].set_ylabel("Predicted Yield (t/ha)")

    plt.show()


def shap_values(model, gdf,pred_cols,value_col,sample_frac=1.0,interactions = False, interaction_features=None, pixel_idx_loc = 0):
    # 1. Initialize the Explainer
    # We use the TreeExplainer, which is lightning-fast for RF and XGBoost
    if sample_frac < 1.0:
        gdf = gdf.sample(frac=sample_frac, random_state=42).copy()
    
    X = gdf[pred_cols]  # Independent variables (predictors) as a DataFrame - this is only from seeding info, not from yld info
    y = gdf[value_col]  # Dependent variable (target)

    explainer = shap.TreeExplainer(model) # 'model' is your trained best model
    shap_values = explainer(X)
    # 2. Calculate SHAP values for the test set
    # (Or pass your entire master_grid if you want to map the explanations)
    
    # Global explanation: Which features drive the model overall across the whole field?
    plt.figure(figsize=(10, 6))
    plt.title("Global Feature Importance (SHAP Summary Plot)")
    shap.summary_plot(shap_values, X)

    plt.figure(figsize=(10,6))
    plt.title("SHAP beeswarm plot")
    shap.plots.beeswarm(shap_values)

    pixel_idx_label = X.index[pixel_idx_loc]
    print(gdf.loc[pixel_idx_label])

    plt.figure(figsize=(10, 6))
    plt.title(f"Explanation for Pixel (Index {pixel_idx_label})")
    shap.plots.waterfall(shap_values[pixel_idx_loc])
    
    if interactions:
        # Computes the full (Samples x Features x Features) matrix
        shap_interaction_values = explainer.shap_interaction_values(X)
        
        # Subset the matrix if specific features are requested
        if interaction_features is not None:
            # Map requested column strings to their numerical index positions
            idx = [X.columns.get_loc(col) for col in interaction_features if col in X.columns]
            
            # Slice the 3D tensor to keep only the requested features on both feature axes
            shap_interaction_values = shap_interaction_values[:, idx, :][:, :, idx]
            X_inter = X[interaction_features]
        else:
            X_inter = X
            
        shap.summary_plot(shap_interaction_values, X_inter)
        

def two_dim_pdp_plots(model, X_train, feat_combos):
    """
    Executes a complete suite of interpretability and diagnostic metrics.
    Assumes a fitted scikit-learn model (like RandomForestRegressor).
    Args:
    trained model
    training data
    2xN numpy array that corresponds to PDPs we want to plot
    """      
    print("Generating 2D Interaction Plots")

    for i in range(0, len(feat_combos)):
        print(f'\n Plotting PDP for {feat_combos[i,0]} vs {feat_combos[i,1]}')
        fig, ax = plt.subplots(figsize=(6,5))
        display = PartialDependenceDisplay.from_estimator(
            model,
            X_train,
            features=[tuple((feat_combos[i,0], feat_combos[i,1]))],
            kind='average',
            contour_kw={'cmap': 'viridis', 'alpha': 0.8},
            ax=ax,
            n_jobs= -1
        )
        
        pdp_vals = display.pd_results[0]['average']
        vmin = np.nanmin(pdp_vals)
        vmax = np.nanmax(pdp_vals)
        step = (vmax-vmin)/10
        cs = display.contours_[0, 0]
        # 2. Generate boundaries and a discrete norm
        bounds = np.arange(vmin, vmax + step, step)
        norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=256)

        # 3. Apply the norm directly to the contour artist
        cs.set_norm(norm)
        cs.set_clim(vmin, vmax)
        
        cax = fig.add_axes([0.99, 0.117, 0.03, 0.81])  
        cbar = fig.colorbar(cs, cax=cax,boundaries = bounds, ticks = bounds, format = '%.2f')
        cbar.set_label("Predicted yield (t/ha)")
        
        ax.set_title(f"PDP: {feat_combos[i,0]} vs {feat_combos[i,1]}")
        plt.tight_layout()
        plt.show()

       
def compare_two_dim_pdp_plots(model_one,model_two, model_names, X_train, feat_combos):
    """
    Executes a complete suite of interpretability and diagnostic metrics.
    Assumes a fitted scikit-learn model (like RandomForestRegressor).
    Args:
    trained model
    training data
    2xN numpy array that corresponds to PDPs we want to plot
    """      
    print("Generating 2D Interaction Plots")

    for i in range(0, len(feat_combos)):
        print(f'\n Plotting PDP for {feat_combos[i,0]} vs {feat_combos[i,1]}')
        
        fig, ax = plt.subplots(1,2,figsize=(10,4))
        
        display_one = PartialDependenceDisplay.from_estimator(
            model_one,
            X_train,
            features=[tuple((feat_combos[i,0], feat_combos[i,1]))],
            kind='average',
            contour_kw={'cmap': 'viridis', 'alpha': 0.8},
            ax=ax[0],
            n_jobs= -1
        )
    
        display_two = PartialDependenceDisplay.from_estimator(
            model_two,
            X_train,
            features=[tuple((feat_combos[i,0], feat_combos[i,1]))],
            kind='average',
            contour_kw={'cmap': 'viridis', 'alpha': 0.8},
            ax=ax[1],
            n_jobs= -1
        )
    
        pdp_vals = display_one.pd_results[0]['average']
        vmin = np.nanmin(pdp_vals)
        vmax = np.nanmax(pdp_vals)
        step = (vmax-vmin)/10
        cs = display_one.contours_[0, 0]
        # 2. Generate boundaries and a discrete norm
        bounds = np.arange(vmin, vmax + step, step)
        norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=256)

        # 3. Apply the norm directly to the contour artist
        cs.set_norm(norm)
        cs.set_clim(vmin, vmax)
        
        cax = fig.add_axes([0.99, 0.16, 0.03, 0.72])  
        cbar = fig.colorbar(cs, cax=cax,boundaries = bounds, ticks = bounds, format = '%.2f')
        cbar.set_label("Predicted yield (t/ha)")
        
        ax[0].set_title(f"{model_names[0]}: {feat_combos[i,0]} vs {feat_combos[i,1]}")
        ax[1].set_title(f"{model_names[1]}: {feat_combos[i,0]} vs {feat_combos[i,1]}")

        plt.tight_layout()
        plt.show()


def plot_comparative_parity(model_results_list):
    n_models = len(model_results_list)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), sharey=True, sharex=True)
    if n_models == 1: axes = [axes]
    
    for ax, res in zip(axes, model_results_list):
        y = res['y_true']
        preds = res['y_pred']
        
        ax.scatter(y, preds, alpha=0.3, s=5, c='blue')
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        ax.set_xlabel('Actual Yield')
        ax.set_title(f"Parity: {res['model_name']}")
        
    axes[0].set_ylabel('Predicted Yield (Out-of-Fold)')
    plt.tight_layout()
    plt.show()


def plot_comparative_residuals(model_results_list):
    n_models = len(model_results_list)
    fig, axes = plt.subplots(1, n_models, figsize=(10, 4), sharey=True, sharex=True)
    if n_models == 1: axes = [axes]
    
    for ax, res in zip(axes, model_results_list):
        y = res['y_true']
        residuals = y - res['y_pred']
        
        ax.scatter(res['y_pred'], residuals, alpha=0.3, s=5, c='blue')
        ax.axhline(0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Yield')
        ax.set_title(f"Residuals: {res['model_name']}")
        
    axes[0].set_ylabel('Residual Error (Actual - Predicted)')
    plt.tight_layout()
    plt.show()


def plot_comparative_importances(model_results_list, top_n=15):
    # Consolidate importances into a single DataFrame for grouped bar plotting
    df_plot = pd.DataFrame()
    for res in model_results_list:
        temp_df = pd.DataFrame({
            'Feature': res['features'], 
            res['model_name']: res['importances']
        }).set_index('Feature')
        
        if df_plot.empty:
            df_plot = temp_df
        else:
            df_plot = df_plot.join(temp_df, how='outer')
            
    # Sort by the average importance across all models and select top N
    df_plot['mean_importance'] = df_plot.mean(axis=1)
    df_plot = df_plot.sort_values('mean_importance', ascending=False).drop(columns=['mean_importance']).head(top_n)
    
    # Plot grouped bar chart
    ax = df_plot.plot(kind='bar', figsize=(10, 4), width=0.8)
    ax.set_ylabel("Permutation Importance")
    ax.set_title("Comparative Feature Importances")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

