# preprocess_drought_data.py
import xarray as xr
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import gc  # Garbage collection

def preprocess_drought_data(netcdf_file, output_file, start_year=2010, end_year=2022, debug=True):
    print(f"Loading NetCDF data from {netcdf_file} (years {start_year}-{end_year})...")

    try:
        # First, examine the file structure without loading all data
        with xr.open_dataset(netcdf_file, engine='netcdf4') as ds:
            if debug:
                print("\nDataset information:")
                print(ds.info())

            # Get coordinate information
            time_values = ds.time.values

            # Identify which years to process
            time_years = pd.DatetimeIndex(time_values).year
            time_indices = np.where((time_years >= start_year) & (time_years <= end_year))[0]

            if len(time_indices) == 0:
                print(f"No data found for years {start_year}-{end_year}")
                print(f"Available years: {min(time_years)}-{max(time_years)}")
                return

            print(f"Processing {len(time_indices)} months from {start_year} to {end_year}")

            # Get spatial dimensions
            if 'lat' in ds and 'lon' in ds:
                # Direct lat/lon
                if ds.lat.dims == ds.lon.dims and len(ds.lat.dims) == 2:
                    # 2D coordinate arrays (common in some models)
                    print("Using 2D coordinate arrays")
                    # We'll handle this differently below
                else:
                    # 1D coordinate arrays
                    latitudes = ds.lat.values
                    longitudes = ds.lon.values
            else:
                print("Coordinates stored in easting/northing with separate lat/lon variables")

        # Memory-efficient processing approach:
        # 1. Load only the time period we need
        # 2. Process data in spatial chunks
        # 3. Calculate the most relevant drought metrics

        # Calculate in chunks to avoid memory issues
        chunk_size = 50  # Adjust based on your available memory

        # First approach: try to load just the years we want
        ds = xr.open_dataset(netcdf_file, engine='netcdf4')

        # Subset to our time period
        ds_subset = ds.isel(time=time_indices)

        # Free up memory
        ds.close()
        del ds
        gc.collect()

        print(f"Processing subset of data ({len(time_indices)} time steps)")

        # Process the SMI data to calculate drought metrics
        smi_var = 'SMI'

        # Get dimensions for chunking
        if 'northing' in ds_subset.dims and 'easting' in ds_subset.dims:
            # Using easting/northing coordinates
            y_dim = 'northing'
            x_dim = 'easting'
            n_y = ds_subset.dims[y_dim]
            n_x = ds_subset.dims[x_dim]
        else:
            # Using direct lat/lon
            y_dim = 'lat'
            x_dim = 'lon'
            n_y = ds_subset.dims[y_dim]
            n_x = ds_subset.dims[x_dim]

        print(f"Processing grid of size {n_y}×{n_x}")

        # Correct thresholds from German drought monitor documentation
        thresholds = {
            'unusual_dryness': {'lower': 0.2, 'upper': 0.3},  # SMI 0.20 - 0.30 = ungewöhnliche Trockenheit
            'moderate_drought': {'lower': 0.1, 'upper': 0.2},  # SMI 0.10 - 0.20 = moderate Dürre
            'severe_drought': {'lower': 0.05, 'upper': 0.1},  # SMI 0.05 - 0.10 = schwere Dürre
            'extreme_drought': {'lower': 0.02, 'upper': 0.05},  # SMI 0.02 - 0.05 = extreme Dürre
            'exceptional_drought': {'lower': 0.0, 'upper': 0.02}  # SMI 0.00 - 0.02 = außergewöhnliche Dürre
        }

        results = []

        # Process in chunks to conserve memory
        for y_start in range(0, n_y, chunk_size):
            y_end = min(y_start + chunk_size, n_y)

            for x_start in range(0, n_x, chunk_size):
                x_end = min(x_start + chunk_size, n_x)

                print(f"Processing chunk: rows {y_start}-{y_end}, cols {x_start}-{x_end}")

                # Select spatial chunk
                chunk = ds_subset.isel({y_dim: slice(y_start, y_end), x_dim: slice(x_start, x_end)})

                # Get lat/lon for this chunk
                lats = chunk.lat.values
                lons = chunk.lon.values

                # Calculate drought statistics for each cell in this chunk
                smi_values = chunk[smi_var].values

                # Iterate through spatial points in this chunk
                for y_idx in range(lats.shape[0]):
                    for x_idx in range(lats.shape[1]):
                        # Get lat/lon
                        lat = lats[y_idx, x_idx]
                        lon = lons[y_idx, x_idx]

                        # Get SMI time series for this location
                        smi_ts = smi_values[:, y_idx, x_idx]

                        # Skip if no valid data
                        if np.all(np.isnan(smi_ts)):
                            continue

                        # Calculate key drought metrics
                        # 1. Average SMI value
                        avg_smi = np.nanmean(smi_ts)

                        # 2. Drought frequency (% of time below each threshold)
                        # Calculate drought percentages with correct thresholds
                        unusual_dry_pct = np.sum((smi_ts >= 0.2) & (smi_ts < 0.3)) / len(smi_ts) * 100
                        moderate_drought_pct = np.sum((smi_ts >= 0.1) & (smi_ts < 0.2)) / len(smi_ts) * 100
                        severe_drought_pct = np.sum((smi_ts >= 0.05) & (smi_ts < 0.1)) / len(smi_ts) * 100
                        extreme_drought_pct = np.sum((smi_ts >= 0.02) & (smi_ts < 0.05)) / len(smi_ts) * 100
                        exceptional_drought_pct = np.sum(smi_ts < 0.02) / len(smi_ts) * 100

                        # Total drought percentage (SMI < 0.2) - all drought categories combined
                        total_drought_pct = np.sum(smi_ts < 0.2) / len(smi_ts) * 100

                        # 4. Recent trend (compare last year to average)
                        if len(smi_ts) >= 12:
                            recent_smi = np.nanmean(smi_ts[-12:])
                            historical_smi = np.nanmean(smi_ts[:-12])
                            trend = recent_smi - historical_smi
                        else:
                            trend = 0

                        # 5. Drought intensity (average SMI during drought periods)
                        drought_periods = smi_ts[smi_ts <= 0.2]
                        if len(drought_periods) > 0:
                            drought_intensity = np.nanmean(drought_periods)
                        else:
                            drought_intensity = np.nan

                        # Calculate drought resistance score based on frequency and intensity
                        # Base score: Higher average SMI = better resistance
                        base_score = avg_smi * 100  # SMI range is 0-1, convert to 0-100

                        # Penalty for drought frequency
                        # Weighted by severity: exceptional droughts impact score more
                        drought_penalty = (
                                moderate_drought_pct * 0.5 +
                                severe_drought_pct * 1.0 +
                                extreme_drought_pct * 1.5 +
                                exceptional_drought_pct * 2.0
                        )

                        # Adjust for trend - worsening conditions reduce score
                        trend_factor = max(-20, min(20, trend * 100))  # Scale trend impact

                        # Final score calculation
                        drought_resistance = max(0, min(100, base_score - drought_penalty + trend_factor))

                        # Store results
                        results.append({
                            'lat': lat,
                            'lon': lon,
                            'avg_smi': avg_smi,
                            'total_drought_pct': total_drought_pct,
                            'unusual_dry_pct': unusual_dry_pct,
                            'moderate_drought_pct': moderate_drought_pct,
                            'severe_drought_pct': severe_drought_pct,
                            'extreme_drought_pct': extreme_drought_pct,
                            'exceptional_drought_pct': exceptional_drought_pct,
                            'drought_intensity': drought_intensity,
                            'trend': trend,
                            'drought_resistance': drought_resistance
                        })

                # Clear chunk from memory
                del chunk
                gc.collect()

        # Convert to DataFrame and save
        df = pd.DataFrame(results)

        # Save to disk
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} data points to {output_file}")

        # Create a simple visualization if debug is enabled
        if debug and len(df) > 0:
            print("Creating visualization...")
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(df['lon'], df['lat'], c=df['drought_resistance'],
                                  cmap='RdYlGn', s=5, alpha=0.7)
            plt.colorbar(scatter, label='Drought Resistance Score')
            plt.title(f'Drought Resistance Score ({start_year}-{end_year})')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.savefig(os.path.splitext(output_file)[0] + '_map.png', dpi=150)
            print(f"Created map at {os.path.splitext(output_file)[0] + '_map.png'}")

            # Create additional visualization for drought categories
            plt.figure(figsize=(12, 8))
            categories = ['unusual_dry_pct', 'moderate_drought_pct',
                          'severe_drought_pct', 'extreme_drought_pct', 'exceptional_drought_pct']
            avg_values = [df[cat].mean() for cat in categories]

            plt.bar(range(len(categories)), avg_values)
            plt.xticks(range(len(categories)), [c.replace('_pct', '').replace('_', ' ') for c in categories],
                       rotation=45)
            plt.title(f'Average Drought Category Distribution ({start_year}-{end_year})')
            plt.ylabel('Percentage of time')
            plt.tight_layout()
            plt.savefig(os.path.splitext(output_file)[0] + '_categories.png', dpi=150)

        # Create simplified spatial grid for fast lookup
        grid_file = os.path.splitext(output_file)[0] + '_grid.pkl'
        create_spatial_grid(df, grid_file)

    except Exception as e:
        import traceback
        print(f"Error processing data: {e}")
        traceback.print_exc()

    finally:
        # Ensure proper cleanup
        try:
            ds_subset.close()
        except:
            pass
        gc.collect()


def create_spatial_grid(df, output_file):
    try:
        # Create KD-Tree for fast spatial lookups
        coords = df[['lat', 'lon']].values
        tree = cKDTree(coords)

        # Save minimal dataset for lookup
        lookup_df = df[['lat', 'lon', 'drought_resistance', 'avg_smi', 'total_drought_pct']].copy()

        # Save tree and data
        import pickle
        with open(output_file, 'wb') as f:
            pickle.dump({'tree': tree, 'data': lookup_df}, f)

        print(f"Spatial grid saved to {output_file}")
    except Exception as e:
        print(f"Error creating spatial grid: {e}")


if __name__ == "__main__":
    netcdf_file = "XXXX/SMI_Gesamtboden_monatlich.nc"
    output_file = "XXXX/drought_resistance_data.csv"

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Process only recent years to reduce memory usage
    # For a first test, just process 5 years
    preprocess_drought_data(netcdf_file, output_file, start_year=2018, end_year=2022, debug=True)