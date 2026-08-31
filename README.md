# Twitch Viewership & Steam Player Dynamics: Granger Causality Analysis (2019-2021)

This repository contains the source code, datasets, and visualizations for a DSCI 550 project examining the statistical relationship between Twitch viewership and Steam player counts. 

## 📊 Project Overview

The primary objective of this project is to determine if a predictive relationship exists between a game's popularity on Twitch (viewership) and its active player base on Steam (concurrent players) between the years 2019 and 2021. 

By applying time-series statistical methods—specifically Vector Autoregression (VAR) and Granger Causality tests—this analysis explores whether spikes in live streaming viewership lead to increases in game adoption and retention, or vice-versa. 

## 📁 Repository Structure

* **`VAR Results/`**: Output data and statistical summaries from the Vector Autoregression models.
* **`cleaned_data/`**: Processed and normalized datasets ready for statistical modeling.
* **`codes/`**: Source code for data pipelines and modeling:
  * `VAR.py` - Vector Autoregression model implementation.
  * `data_cleaner.py` - Data ingestion, cleaning, and preprocessing.
  * `data_visualization.py` - Visualization generators.
  * `granger_causality.py` - Statistical testing for Granger causality.
  * `merge_steam_twitch.py` - Scripts to join and align Steam and Twitch datasets.
* **`overall_trends_visualizations/`**: Macro-level trend charts across the 2019-2021 timeline.
* **`raw_data/`**: Original, unprocessed data dumps.
* **`scatter_visualizations/`**: Scatter plots exploring statistical correlations.
* **`top_games_visualizations/`**: Charts highlighting specific high-performing titles.

## 🛠️ Tech Stack & Methods
* **Languages:** Python (Pandas, NumPy)
* **Statistical Modeling:** Statsmodels (VAR, Granger Causality)
* **Data Visualization:** Matplotlib, Seaborn

## 🚀 How to Run

1. **Clone the repository:**
```bash
git clone [https://github.com/glenjpdavid/dsci550-twitch-steam-project.git](https://github.com/glenjpdavid/dsci550-twitch-steam-project.git)
cd dsci550-twitch-steam-project
```

2. **Prepare the Data:**
Run `codes/data_cleaner.py` and `codes/merge_steam_twitch.py` to process the raw datasets.

3. **Execute Models & Tests:**
Run `codes/VAR.py` followed by `codes/granger_causality.py` to generate the statistical outputs. 

4. **Generate Visuals:**
Run `codes/data_visualization.py` to output the charts to their respective directories.

## ✍️ Authors & Acknowledgements

* **Glen J. David** - Data Scientist & ML Engineer | [LinkedIn](https://linkedin.com/in/glenjpdavid)
* **Sim Wang** - Code Contributor & Project Collaborator
* **Lizzy Brunn** - Project Collaborator
* **Vy Nguyen** - Project Collaborator
