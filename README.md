# DroughtRiskMetric-Germany

### Project Overview
The Drought Resistance Analysis Tool is a Python-based data processing pipeline designed to analyze drought conditions using soil moisture index (SMI) data from the German Drought Monitor. 
Droughts pose significant challenges to ecosystems, agriculture, and water resource management, with increasing frequency and severity due to climate change. The Soil Moisture Index (SMI), a standardized metric ranging from 0 to 1, quantifies soil moisture conditions, where lower values indicate drier conditions. The German Drought Monitor uses SMI thresholds to categorize drought severity, ranging from "unusual dryness" (SMI 0.2–0.3) to "exceptional drought" (SMI < 0.02). This project was motivated by the need to process and analyze large NetCDF datasets containing SMI data to derive meaningful drought metrics, such as frequency, intensity, and resistance scores, while addressing computational challenges like memory efficiency and scalability.

### Data Processing:

Loads and processes NetCDF files containing SMI data using xarray for efficient handling of multidimensional arrays.
Implements chunk-based processing to manage large datasets, reducing memory usage and preventing crashes on resource-constrained systems.
Subsets data by user-defined time periods (default: 2010–2022) to focus on relevant temporal ranges.


### Drought Metrics Calculation:

The drought resistance score is calculated using a weighted formula that balances average SMI, drought frequency penalties (with higher weights for severe categories), and recent trends, ensuring a comprehensive assessment of drought resilience.

Computes critical drought metrics for each spatial point, including:

- Average SMI: Mean soil moisture index over the specified period.
- Drought Frequency: Percentage of time spent in each drought category (unusual dryness, moderate, severe, extreme, and exceptional drought).
- Drought Intensity: Average SMI during drought periods (SMI < 0.2).
- Recent Trend: Comparison of the last year's SMI to the historical average.
- Drought Resistance Score: A composite score (0–100) based on average SMI, drought frequency, and recent trends, providing a measure of resilience to drought conditions.


### Spatial Analysis:

Supports both lat/lon and easting/northing coordinate systems, accommodating various NetCDF data formats.
Creates a spatial lookup grid using a KD-Tree (scipy.spatial.cKDTree) for fast nearest-neighbor queries, saved as a pickle file for efficient reuse.


### Visualization:

Generates two types of visualizations when debug mode is enabled:
A scatter plot map showing drought resistance scores across geographic locations.
A bar plot illustrating the average distribution of time spent in each drought category.

### Technical Implementation
The tool leverages the following libraries:

- xarray: For reading and manipulating NetCDF files with multidimensional arrays.
- pandas: For structuring and saving results as CSV files.
- numpy: For numerical computations, including statistical metrics.
- matplotlib: For generating visualizations of drought resistance and category distributions.
- scipy.spatial.cKDTree: For creating spatial lookup grids.
- os and gc: For file handling and memory management.

The processing pipeline is optimized for memory efficiency by:

- Loading only the required time period from the NetCDF file.
- Processing spatial data in chunks (default size: 50×50 grid cells).
- Releasing memory using garbage collection after each chunk.


### Installation and Usage
Prerequisites:

- Python 3.8+
-  Required libraries: xarray, pandas, numpy, matplotlib, scipy
- Install dependencies using: ```pip install xarray pandas numpy matplotlib scipy```


Running the Tool:

Place your NetCDF file (e.g., SMI_Gesamtboden_monatlich.nc) in the project directory or specify its path.
Run the script with default or custom parameters: ```python drought-pre-V2.py```


Outputs include:
- A CSV file (drought_resistance_data.csv) with drought metrics for each spatial point.
- A pickle file (drought_resistance_data_grid.pkl) for spatial lookups.
- Two PNG visualizations (if debug=True): a drought resistance map and a drought category distribution plot.


Example
```
preprocess_drought_data(
    netcdf_file="data/SMI_Gesamtboden_monatlich.nc",
    output_file="output/drought_resistance_data.csv",
    start_year=2018,
    end_year=2022,
    debug=True
)
```
