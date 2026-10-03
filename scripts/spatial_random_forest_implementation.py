
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import matplotlib.ticker as ticker
import seaborn as sns
import matplotlib.pyplot as plt
# Import PyGRF components NB the MODIFIED one is MY OWN implementation, adapted to support cuML
from PyGRF import PyGRFBuilder
from PyGRF import search_bw_lw_ISA  # Incremental Spatial Autocorrelation tool
from shapely import Point
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error as rmse
import numpy as np



def prep_dataset(full_dataset, yield_colname, pred_cols, NON_FEATURE_COLS):

    # To use full DF
    # Below only needed if not doing the 5-fold initial split above
    x_coords = full_dataset.geometry.centroid.x
    y_coords = full_dataset.geometry.centroid.y

    full_dataset['x'] = x_coords
    full_dataset['y'] = y_coords

    pred_cols = [
        col for col in full_dataset.columns
        if col not in NON_FEATURE_COLS
    ]
    print(f"Predictor columns ({len(pred_cols)}): {pred_cols}")

    X = full_dataset[pred_cols]  # Independent variables (predictors) as a DataFrame - this is only from seeding info, not from yld info
    y = full_dataset[yield_colname]  # Dependent variable (target)
    # Train/Test Split (Ensuring coordinates split identically to features)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    
    training_coords = X_train[['x', 'y']]
    testing_coords = X_test[['x', 'y']]
    
    return X_train, X_test, y_train, y_test, training_coords, testing_coords


def spatial_parameters(y_train, X_train):
    # Optimize Spatial Hyperparameters Automatically
    print("Calculating ideal bandwidth via Incremental Spatial Autocorrelation...")
    # This uses Moran's I to find the neighborhood size where spatial patterns are strongest
    bandwidth, morans_i, p_value = search_bw_lw_ISA(
        y=y_train, 
        coords=X_train[['x', 'y']],
        bw_min=5, 
        bw_max=200, 
        step=5
    )
    local_weight = max(0, morans_i) # Leverage Moran's I as the structural local weight mixing parameter

    print(f"Optimal Bandwidth: {bandwidth} nearest neighbors")
    print(f"Calculated Local Weight: {local_weight:.4f} (Based on Moran's I: {morans_i:.4f})")

    return bandwidth, local_weight


def train_grf_model(X_train, y_train,X_test,y_test, training_coords, testing_coords, bandwidth, local_weight):
    
    # Initialize and Fit the PyGRF Model
    pygrf_model = PyGRFBuilder(
        n_estimators=200,        # Number of trees in individual forests
        band_width=bandwidth,    # Size of local neighborhoods
        train_weighted=True,     # Spatially weight local samples during training
        predict_weighted=True,   # Apply spatial weights to local prediction blending
        resampled=True,          # Local sample expansion to stabilize sparse cells
        random_state=42,
        n_jobs = -1
    )

    print("\nTraining local models across coordinates...")
    pygrf_model.fit(X_train, y_train, training_coords)

    # 5. Predict and Evaluate
    # Predict returns: combined prediction, global-only prediction, and local-only prediction
    predict_combined, predict_global, predict_local = pygrf_model.predict(
        X_test, 
        testing_coords,
        local_weight=local_weight
    )

    print("\n--- Model Performance Evaluation ---")
    print(f"PyGRF Combined R²: {r2_score(y_test, predict_combined):.4f}")
    print(f"Global-Only Model R²: {r2_score(y_test, predict_global):.4f}")
    print(f"Combined RMSE: {rmse(y_test, predict_combined):.4f}")
    
    # 6. Extract Local Feature Importance to map non-stationarity
    local_importance_df = pygrf_model.get_local_feature_importance()

    print("\nLocal Feature Importance Matrix (Snippet for first 5 coordinate zones):")
    # Each row represents the importance profile belonging to that unique point's neighborhood model
    print(local_importance_df.head())


    global_importance_df = pd.DataFrame()
    global_importance_df["Quantity"] = ['Sum','Minimum','Maximum','Mean','Standard Deviation']

    for col in local_importance_df.columns:
        global_importance_df[col] = [local_importance_df[col].sum(),local_importance_df[col].min(),local_importance_df[col].max(), local_importance_df[col].mean(), local_importance_df[col].std()]

    print("\nGlobal Feature Importance Matrix:")
    print(global_importance_df.head())

    return pygrf_model, predict_combined, predict_global, local_importance_df, global_importance_df



def plot_results(y_test, predict_combined, testing_coords, training_coords, pygrf_model, pred_cols, local_importance_df):
    # --- VISUALIZATION WORKFLOW ---

    # 5. Mapping Residuals (on the Test Set)
    # Calculate local residuals (Actual - Predicted)
    test_residuals = y_test - predict_combined

    # Create a GeoDataFrame linking coordinates to residuals
    gdf_test = gpd.GeoDataFrame(
        {'Residuals': test_residuals},
        geometry=[Point(xy) for xy in zip(testing_coords['x'], testing_coords['y'])]
    )

    # 6. Mapping Local Feature Importance (on the Train Set)
    # Extract the local importance matrix (1 row per training coordinate)

    # Bind importance values back to the training coordinates
    gdf_train = gpd.GeoDataFrame(
        local_importance_df,
        geometry=[Point(xy) for xy in zip(training_coords['x'], training_coords['y'])]
    )


    global_feat_imp = pygrf_model.global_model.feature_importances_


    # 1. Filter the arrays
    f_floats = global_feat_imp
    f_strings = np.array(pred_cols)

    # 2. Sort the remaining data in descending order
    sort_idx = np.argsort(f_floats)[::-1]
    s_floats, s_strings = f_floats[sort_idx], f_strings[sort_idx]

    # 3. Apply the top 10 (with ties) threshold and plot
    if len(s_floats) > 0:
        threshold = s_floats[min(9, len(s_floats) - 1)]
        plot_mask = s_floats >= threshold
        
        plt.figure(figsize=(10, 6))
        plt.bar(s_strings[plot_mask], s_floats[plot_mask])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
    return gdf_train, gdf_test



def plot_local_r2_error(pygrf_model, training_coords):
    local_r2 = pygrf_model.get_local_R2()

    local_r2_plot_df = gpd.GeoDataFrame(
        local_r2,
        geometry=[Point(xy) for xy in zip(training_coords['x'], training_coords['y'])]
    )

    # 7. Plotting the Results
    fig, ax = plt.subplots(1, 1, figsize=(6,4))

    # Plot 1: Spatial Residuals (Are errors randomly distributed or clustered?)
    local_r2_plot_df.plot(column='local R2', cmap='RdYlGn', legend=True, ax=ax, markersize=3)
    ax.set_title("Local model R2 score")
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4)) 
    plt.tight_layout()
    plt.show()



def plot_residuals(gdf_test):
    # 7. Plotting the Results
    fig, ax = plt.subplots(1, 1, figsize=(6,4))

    # Plot 1: Spatial Residuals (Are errors randomly distributed or clustered?)
    gdf_test.plot(column='Residuals', cmap='coolwarm', legend=True, ax=ax, markersize=3)
    ax.set_title("Spatial Residuals (Test Set)")
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.axis('equal')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4)) 
    plt.tight_layout()
    plt.show()



def plot_local_feat_imp(gdf_train, feats):
    l = len(feats)
    fig,ax = plt.subplots(1, l, figsize=(11, 4))
    for i in range(0, l):
        col = feats[i]
        gdf_train.plot(column=col, cmap='viridis', legend=True, ax=ax[i], markersize=3, legend_kwds={'label': 'Gini feature importance'})
        ax[i].set_title(f"Local Feature Importance: {feats[i]}")
        ax[i].set_xlabel("Easting")
        ax[i].set_ylabel("Northing")
        ax[i].xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax[i].yaxis.set_major_locator(ticker.MaxNLocator(nbins=4)) 

    plt.tight_layout()
    plt.show()



def plot_actual_pred_res(predict_combined, testing_coords, y_test):

    predict_df = gpd.GeoDataFrame({
        'Predicted_Yield': predict_combined,
        'geometry': [Point(xy) for xy in zip(testing_coords['x'], testing_coords['y'])]
    })

    test_df = gpd.GeoDataFrame({
        'Actual_Yield': y_test,
        'geometry': [Point(xy) for xy in zip(testing_coords['x'], testing_coords['y'])]
    })



    fig,ax = plt.subplots(1, 3, figsize=(11, 3))
    # Plot 2: Spatial Non-Stationarity of Feature A
    col = 'Predicted_Yield'  # Replace with the actual feature name you want to visualize
    predict_df.plot(column=col, cmap='RdYlGn', legend=True, ax=ax[1], markersize=3, vmin = 0.3, vmax = 2.85, legend_kwds = {'label' : 'Yield (t/ha)'})
    ax[1].set_title(f"Predicted yield")
    ax[1].set_xlabel("Easting")
    ax[1].set_ylabel("Northing")
    ax[1].xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    ax[1].yaxis.set_major_locator(ticker.MaxNLocator(nbins=3)) 

    # Plot 3: Spatial Non-Stationarity of Feature B
    col = 'Actual_Yield'  # Replace with the actual feature name you want to visualize
    test_df.plot(column=col, cmap='RdYlGn',  legend=True, ax=ax[0], markersize=3, vmin = 0.3, vmax = 2.85, legend_kwds = {'label' : 'Yield (t/ha)'})
    ax[0].set_title(f"Actual yield")
    ax[0].set_xlabel("Easting")
    ax[0].set_ylabel("Northing")
    ax[0].xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    ax[0].yaxis.set_major_locator(ticker.MaxNLocator(nbins=3)) 

    gdf_test.plot(column='Residuals', cmap='coolwarm', legend=True, ax=ax[2], markersize=3, legend_kwds = {'label' : 'Residuals (t/ha)'})
    ax[2].set_title("Residuals (Test Set)")
    ax[2].set_xlabel("Easting")
    ax[2].set_ylabel("Northing")
    ax[2].xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    ax[2].yaxis.set_major_locator(ticker.MaxNLocator(nbins=3)) 

    plt.tight_layout()
    plt.show()


def plot_parity_residuals(y_test, predict_combined):

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    sns.scatterplot(x=y_test, y=predict_combined, alpha=0.4, color='blue', ax=ax[0])
    ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax[0].set_title("Actual vs predicted yield")
    ax[0].set_xlabel("Actual yield")
    ax[0].set_ylabel("Predicted yield")

    sns.scatterplot(x=predict_combined, y=test_residuals, alpha=0.4, color='blue', ax=ax[1])
    ax[1].axhline(0, color='red', linestyle='--')
    ax[1].set_title("Residual plot")
    ax[1].set_xlabel("Predicted yield")
    ax[1].set_ylabel("Residuals")


    plt.tight_layout()
    plt.show()