import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import shapely as sp
from sklearn.neighbors import KDTree
from sklearn.preprocessing import OneHotEncoder


def trial_boundaries(df, plots=False, colname = None):
   
    full_poly = df.union_all()
    l = len(full_poly.geoms)

    a = [sp.area(df["geometry"].iloc[x]) for x in range(0, len(df))]

    max_A = max(a)
    max_A_index = a.index(max_A)
    removed = df.iloc[max_A_index:max_A_index+1]
    trial_only = df.drop(index = max_A_index)
    
    b = [sp.area(trial_only["geometry"].iloc[x]) for x in range(0, len(trial_only))]

    if plots:
        fig, ax = plt.subplots(1,3, figsize = (11,3))
        df.plot(column = colname, cmap = 'RdYlGn',edgecolor = 'white', alpha = 0.5, ax = ax[0])
        ax[0].set_xlabel('Easting')
        ax[0].set_ylabel('Northing')
        ax[0].set_title("Original map")
        df.plot(color = 'red', alpha = 0.5, ax = ax[1])
        trial_only.plot(color = 'blue', edgecolor = 'white', alpha = 1, ax = ax[1])
        ax[1].set_xlabel('Easting')
        ax[1].set_ylabel('Northing')
        ax[1].set_title("Trial polygons only")
        removed.plot(color = 'red', alpha = 0.5, ax = ax[2])
        ax[2].set_xlabel('Easting')
        ax[2].set_ylabel('Northing')    
        ax[2].set_title("Removed data")
        plt.suptitle("Area removed from map")
        plt.tight_layout()
        plt.show()
    
    print(f"The original dataset had {len(df)} rows, with a total area of {sum(a)} m^2.")
    print(f"The resulting dataset has {len(trial_only)} rows, with a total area of {sum(b)} m^2.")
    print(f"The resulting dataset has a total area of {sum(b)/10000} ha, which is {sum(b)/sum(a)*100:.2f}% of the original area.")
    
    return trial_only


def remove_boundary_data(input_df, trial_df, buffer_zone = 0, plots= False, colname = None):
    
    # 1. Get the boundary polygon
    b = trial_df.union_all().buffer(1)
    area_bef = sp.Polygon(sp.MultiPoint(input_df.geometry.tolist()).convex_hull).area
    
    # 2. Use gpd.clip instead of .where()
    # This automatically drops anything outside the boundary instantly
    clean_input_data = gpd.clip(input_df, b.buffer(-buffer_zone))
    
    area_aft = clean_input_data.unary_union.convex_hull.area
    
    # 3. Find the removed points using the index
    removed = input_df.loc[~input_df.index.isin(clean_input_data.index)]
        
    if plots:
        fig, ax = plt.subplots(1,3, figsize = (11,3))
        input_df.plot(color = 'blue', alpha = 0.5, ax = ax[0])
        input_df.plot(column = colname, cmap = 'RdYlGn', alpha = 0.5, ax = ax[0], markersize = 4)
        ax[0].axis('equal')
        ax[0].set_xlabel('Easting')
        ax[0].set_ylabel('Northing')
        ax[0].set_title("Original data")
        
        input_df.plot(color = 'blue', alpha = 0.5, ax = ax[1])
        clean_input_data.plot(column = colname, cmap = 'RdYlGn', alpha = 0.5, ax = ax[1], markersize = 4)
        ax[1].axis('equal')
        ax[1].set_title("Remaining data")
        ax[1].set_xlabel('Easting')
        ax[1].set_ylabel('Northing')
        
        removed.plot(column = colname, cmap = 'RdYlGn', alpha = 0.5, ax = ax[2], markersize = 4)
        ax[2].axis('equal')
        ax[2].set_title("Removed data")
        ax[2].set_xlabel('Easting')
        ax[2].set_ylabel('Northing')
        
        ax[0].xaxis.set_major_locator(ticker.MaxNLocator(nbins=(5)))
        ax[0].yaxis.set_major_locator(ticker.MaxNLocator(nbins=(5)))
        ax[1].xaxis.set_major_locator(ticker.MaxNLocator(nbins=(5)))
        ax[1].yaxis.set_major_locator(ticker.MaxNLocator(nbins=(5)))
        ax[2].xaxis.set_major_locator(ticker.MaxNLocator(nbins=(5)))
        ax[2].yaxis.set_major_locator(ticker.MaxNLocator(nbins=(5)))
        
        plt.suptitle("The data remaining after filtering by the trial boundaries.")        
        plt.tight_layout()
        plt.show()

    print(f"The original dataset had {len(input_df)} rows, with a total area of {area_bef} m^2.")
    print(f"The resulting dataset has {len(clean_input_data)} rows, with a total earea of {area_aft} m^2.")
    print(f"The resulting dataset has a total area of {area_aft/10000} ha, which is {area_aft/area_bef*100:.2f}% of the original area.")
    print(f"The dataframe length is which is {len(clean_input_data)/len(input_df)*100:.2f}% of the original area")
    
    return clean_input_data, removed


def filter_parameter(input_df,seed_colname, parameter_colname, method, lower_threshold = 0.10, upper_threshold = 0.90, plots = True):

    input_df = input_df.dropna(subset=[seed_colname, parameter_colname])

    if method == 'percentile':
        print(f"Filtering {parameter_colname} to the range between the {lower_threshold*100:.0f}th and {upper_threshold*100:.0f}th percentiles.")
        lower = input_df[parameter_colname].quantile(lower_threshold)
        upper = input_df[parameter_colname].quantile(upper_threshold)
    elif method == 'bounds':
        print(f"Filtering {parameter_colname} to the range between {lower_threshold} and {upper_threshold}.")
        lower = lower_threshold
        upper = upper_threshold
    elif method == 'SD':
        print(f"Filtering {parameter_colname} to the range between -{lower_threshold} and {upper_threshold} standard deviations from the mean.")
        mean = input_df[parameter_colname].mean()
        std = input_df[parameter_colname].std()
        lower = mean - lower_threshold * std
        upper = mean + upper_threshold * std
    else:
        raise ValueError("Method must be 'percentile', 'bounds', or 'SD'.")
    
    # Keep points strictly between these two values
    try:
        filtered_input_df = input_df[(input_df[parameter_colname] >= lower) & (input_df[parameter_colname] <= upper)]
    except Exception as e:
        print(f"Error occurred while filtering: {e}")


    if plots:
        fig, ax = plt.subplots(2,2,figsize=(6, 4))
        counts, bins, patches = ax[0,0].hist(input_df[parameter_colname], bins=50)
        ax[0,0].set_xlabel(parameter_colname)
        ax[0,0].set_ylabel('Frequency')
        ax[0,0].set_title('Distribution of ' + parameter_colname)

        # Add value counts on top of each bar
        for count, patch in zip(counts, patches):
            height = patch.get_height()
            if height > 0:  # Only label non-zero bars
                ax[0,0].text(patch.get_x() + patch.get_width()/2., height,
                        f'{int(count)}', ha='center', va='bottom', fontsize=8)

        counts, bins, patches = ax[0,1].hist(input_df[seed_colname], bins=50)
        ax[0,1].set_xlabel(seed_colname)
        ax[0,1].set_ylabel('Frequency')
        ax[0,1].set_title(f'Distribution of {seed_colname}')

        # Add value counts on top of each bar
        for count, patch in zip(counts, patches):
            height = patch.get_height()
            if height > 0:  # Only label non-zero bars
                ax[0,1].text(patch.get_x() + patch.get_width()/2., height,
                        f'{int(count)}', ha='center', va='bottom', fontsize=8)
                
        counts, bins, patches = ax[1,0].hist(filtered_input_df[parameter_colname], bins=50)
        ax[1,0].set_xlabel(parameter_colname)
        ax[1,0].set_ylabel('Frequency')
        ax[1,0].set_title('Distribution of ' + parameter_colname)

        # Add value counts on top of each bar
        for count, patch in zip(counts, patches):
            height = patch.get_height()
            if height > 0:  # Only label non-zero bars
                ax[1,0].text(patch.get_x() + patch.get_width()/2., height,
                        f'{int(count)}', ha='center', va='bottom', fontsize=8)

        counts, bins, patches = ax[1,1].hist(filtered_input_df[seed_colname], bins=50)
        ax[1,1].set_xlabel(seed_colname)
        ax[1,1].set_ylabel('Frequency')
        ax[1,1].set_title(f'Distribution of {seed_colname}')

        # Add value counts on top of each bar
        for count, patch in zip(counts, patches):
            height = patch.get_height()
            if height > 0:  # Only label non-zero bars
                ax[1,1].text(patch.get_x() + patch.get_width()/2., height,
                        f'{int(count)}', ha='center', va='bottom', fontsize=8)
                
        plt.tight_layout()
        plt.show()

    print(f"Original dataframe had {len(input_df)} points. Filtered dataframe has {len(filtered_input_df)} points, which is {len(filtered_input_df)/len(input_df)*100:.2f}% of the original.")
    print(f"{parameter_colname} range: {filtered_input_df[parameter_colname].min()} to {filtered_input_df[parameter_colname].max()}")
    filtered_input_df[seed_colname].describe()
    
    return filtered_input_df


def remove_local_outliers(gdf, value_col, crit='dist', s=15, sd_threshold=2.0, plot_removed=True, plot_example=True):
    """
    Removes spatial anomalies by comparing each point to its local neighborhood.
    """
    print(f"Running Local Outlier Detection on '{value_col}' (crit={crit}, param={s}, SD={sd_threshold})...")
    
    # 1. Extract coordinates and values
    coords = np.column_stack((gdf.geometry.x, gdf.geometry.y))
    values = gdf[value_col].values
    tree = KDTree(coords)

    # 2. Arrays to hold our calculated local statistics
    local_means = np.zeros(len(gdf))
    local_sds = np.zeros(len(gdf))
    neighbor_indices_list = []

    # 3. Query the tree based on the chosen criteria
    if crit == 'neighbours':
        _, indices = tree.query(coords, k=s+1)
        
        # In a k-NN query, the point itself is always at index 0. Slice it out.
        neighbor_indices = indices[:, 1:]
        neighbor_values = values[neighbor_indices]
        
        local_means = np.mean(neighbor_values, axis=1)
        local_sds = np.std(neighbor_values, axis=1)
        neighbor_indices_list = neighbor_indices.tolist()

    elif crit == 'dist':
        indices = tree.query_radius(coords, r=s)
        
        # Because different points have different numbers of neighbors within a radius,
        # we must iterate through the ragged array.
        for i, neighbors in enumerate(indices):
            # Filter out the point itself (query_radius does not sort by distance)
            actual_neighbors = neighbors[neighbors != i]
            neighbor_indices_list.append(actual_neighbors.tolist())
            
            if len(actual_neighbors) > 0:
                n_vals = values[actual_neighbors]
                local_means[i] = np.mean(n_vals)
                local_sds[i] = np.std(n_vals)
            else:
                # If no neighbors fall within the radius, it can't be an outlier based on neighbors
                local_means[i] = values[i]
                local_sds[i] = 0.0

    # 4. Define bounds and create the boolean mask
    upper_bound = local_means + (sd_threshold * local_sds)
    lower_bound = local_means - (sd_threshold * local_sds)
    valid_mask = (values <= upper_bound) & (values >= lower_bound)
    
    # Save the neighbor indices so we can plot them later
    gdf["neighbor_indices"] = neighbor_indices_list
    
    # 5. Split the GeoDataFrames
    clean_gdf = gdf[valid_mask].copy()
    outliers_gdf = gdf[~valid_mask].copy()
    
    print(f"Removed {len(outliers_gdf)} local outliers. ({len(clean_gdf)} remaining)")
    
    # 6. Global Visualization
    if plot_removed and len(outliers_gdf) > 0:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        vmax = clean_gdf[value_col].max()
        vmin = clean_gdf[value_col].min()
        
        clean_gdf.plot(column=value_col, cmap='RdYlGn', markersize=4, ax=ax[0], alpha=0.8,vmax = vmax, vmin = vmin, legend=True, legend_kwds = {'label': 'Yield mass (t/ha)'})
        ax[0].set_title(f"Clean yield map")
        ax[0].set_xlabel('Easting')
        ax[0].set_ylabel('Northing')
        ax[0].xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax[0].yaxis.set_major_locator(ticker.MaxNLocator(nbins=5)) 
        ax[0].axis('equal')
        
        clean_gdf.plot(color='lightgrey', markersize=4, ax=ax[1], alpha=0.3)
        outliers_gdf.plot(color='red', markersize=6, ax=ax[1], label='Outlier')
        ax[1].set_title(f"Removed local outliers")
        ax[1].set_xlabel('Easting')
        ax[1].legend()
        ax[1].xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax[1].yaxis.set_major_locator(ticker.MaxNLocator(nbins=5)) 
        ax[1].axis('equal')
        plt.tight_layout()
        plt.show()
        
    # 7. Local Example Visualization
    if plot_example and len(outliers_gdf) > 0:
        # Extract the FIRST outlier using double brackets to keep it as a GeoDataFrame
        example_outlier = outliers_gdf.iloc[[0]] 
        
        # Extract the indices of its neighbors and locate them in the original dataset
        n_idx = example_outlier.iloc[0]["neighbor_indices"]
        example_neighbors = gdf.iloc[n_idx]
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        
        vmax = example_neighbors[value_col].max()
        vmin = example_neighbors[value_col].min()
        
        # Plot A: Spatial Neighborhood Map
        example_neighbors.plot(column=value_col, cmap='RdYlGn', markersize=50, ax=ax[0], alpha=0.8, legend=True, vmin = vmin, vmax = vmax, legend_kwds = {'label': 'Yield mass (t/ha)'})
        example_outlier.plot(column=value_col, cmap='RdYlGn', marker = '*', markersize=100, ax=ax[0], label='Target Outlier', vmin = vmin, vmax = vmax)
        ax[0].set_title('Local Neighbourhood Spatial View')
        ax[0].set_xlabel('Easting')
        ax[0].set_ylabel('Northing')
        ax[0].legend()
        ax[0].xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax[0].yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax[0].axis('equal')

        
        # Plot B: Neighborhood Histogram
        counts, bins, patches = ax[1].hist(example_neighbors[value_col], bins=10, edgecolor='black', alpha=0.7)
        
        # Get the literal scalar value of the outlier for the vertical line
        target_val = example_outlier.iloc[0][value_col]
        
        ax[1].axvline(x=target_val, color="red", linestyle="--", linewidth=2, label=f"Target ({target_val:.2f} t/ha)")
        ax[1].set_xlabel('Yield mass (t/ha)')
        ax[1].set_ylabel('Frequency')
        ax[1].set_title('Neighbour Distribution vs Outlier')
        ax[1].legend()

        # Add value counts on top of each bar
        for count, patch in zip(counts, patches):
            height = patch.get_height()
            if height > 0: 
                ax[1].text(patch.get_x() + patch.get_width()/2., height, f'{int(count)}', ha='center', va='bottom', fontsize=10)
                
        plt.tight_layout()
        plt.show()
        
    # Drop the temporary column before returning to keep the dataframe clean
    clean_gdf = clean_gdf.drop(columns=["neighbor_indices"])
    outliers_gdf = outliers_gdf.drop(columns=["neighbor_indices"])
        
    return clean_gdf, outliers_gdf


def remove_spatial_outliers(gdf, dist, n, plots=False, value_col=None, max_iters=10):
    """
    Iteratively removes spatial outliers. If dropping points exposes new points 
    that now fall below the density threshold, those are dropped in the next pass.
    """
    print(f"--- Starting Spatial Outlier Removal (Radius: {dist}m, Min Neighbors: {n}) ---")
    
    clean_gdf = gdf.copy()
    iteration = 1
    
    while iteration <= max_iters:
        # 1. Extract centroids (safely handles your 10x2m polygons)
        x = clean_gdf.geometry.centroid.x
        y = clean_gdf.geometry.centroid.y
        points = np.column_stack((x, y))
        
        # 2. Build Tree and Query
        tree = KDTree(points)
        all_neighbor_indices = tree.query_radius(points, r=dist)
        
        # 3. Count neighbors
        neighbor_counts = np.array([len(neighbors) for neighbors in all_neighbor_indices])
        
        # 4. Create Mask (+1 because the point counts itself as a neighbor)
        mask_to_keep = neighbor_counts >= (n + 1)
        
        points_dropped_this_round = (~mask_to_keep).sum()
        
        # 5. Break if the map is completely solid and clean
        if points_dropped_this_round == 0:
            print(f"  -> Converged after {iteration-1} iterations. No more outliers found.")
            break
            
        print(f"  -> Iteration {iteration}: Dropped {points_dropped_this_round} points.")
        clean_gdf = clean_gdf[mask_to_keep].copy()
        iteration += 1

    # 6. Isolate the dropped points for plotting
    dropped_gdf = gdf[~gdf.index.isin(clean_gdf.index)].copy()
    print(f"Total points dropped: {len(dropped_gdf)}")
    print("--------------------------------------------------")

    # 7. Visualization
    if plots:
        fig, ax = plt.subplots(1, 3, figsize=(11,3))
        
        # Plot 1: Original
        gdf.plot(column=value_col, cmap="RdYlGn", markersize=4, alpha=0.5, ax=ax[0])
        ax[0].set_title(f"Original Map ({len(gdf):,} points)")
        ax[0].axis('equal')
        
        # Plot 2: What was removed
        gdf.plot(color="lightgrey", alpha=0.3, ax=ax[1])    
        if not dropped_gdf.empty:
            dropped_gdf.plot(column=value_col, cmap="RdYlGn", markersize=8, ax=ax[1])
        ax[1].set_title(f"Dropped Points ({len(dropped_gdf):,})") 
        ax[1].axis('equal')
               
        # Plot 3: Final Clean Result
        clean_gdf.plot(column=value_col, cmap="RdYlGn", markersize=4, alpha=0.5, ax=ax[2])
        ax[2].set_title(f"Final Clean Map ({len(clean_gdf):,})")
        ax[2].axis('equal')
        
        ax[0].xaxis.set_major_locator(ticker.MaxNLocator(nbins=(5)))
        ax[0].yaxis.set_major_locator(ticker.MaxNLocator(nbins=(5)))
        ax[1].xaxis.set_major_locator(ticker.MaxNLocator(nbins=(5)))
        ax[1].yaxis.set_major_locator(ticker.MaxNLocator(nbins=(5)))
        ax[2].xaxis.set_major_locator(ticker.MaxNLocator(nbins=(5)))
        ax[2].yaxis.set_major_locator(ticker.MaxNLocator(nbins=(5)))
        
        plt.suptitle("Iterative Spatial Outlier Removal")
        plt.tight_layout()
        plt.show()
    
    return clean_gdf


def text_encoder(cols, df):
    """
    Function to perform one-hot encoding on non-numeric columns of a DataFrame.
    Args:
        cols (list): List of non-numeric column names to encode.
        df (DataFrame): DataFrame containing the data.
    Returns:
        df (DataFrame): DataFrame with one-hot encoded columns. The original non-numeric columns are dropped.
        names (list): List of names of the new one-hot encoded columns to easily include them in the prediction.
    """
    df = df.dropna(subset=cols)
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    try:
        enc_arr = encoder.fit_transform(df[cols])
        enc_df = gpd.GeoDataFrame(enc_arr, columns=encoder.get_feature_names_out(cols))
        names = enc_df.columns.to_list()
        df = pd.concat([df.drop(columns=cols), enc_df], axis=1)
        print("Remember to include all of the one-hot encoded columns in the prediction!")
    except ValueError as e:
        names = []
        df = df.drop(columns=cols, inplace=True)
        print(f"Error occurred while fitting encoder: {e}")

    return df, names

