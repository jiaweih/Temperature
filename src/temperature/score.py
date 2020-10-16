import numpy as np
import pandas as pd


def scorelator(df, n_samples=10000):
    """Estimate the score for each annual mean temperature.
       The score is estimated as the area between lower bound and the x-axis.
    """
    col_draws = ['draw_{}'.format(i) for i in np.arange(n_samples)]
    score_dict = {}
    # Loop over annual temperature
    for annual_temp in df.annual_temperature.unique():
        # Get draws
        draws = df.loc[df.annual_temperature==annual_temp, col_draws]
        # Return the index of row that corresponds to the minimum mean draw
        # In other words, the row of the daily mean temperature that has the lowest mean value
        min_index = np.argmin(np.mean(draws, axis=1))
        # Shift the draws by getting the difference of draws and minimum mean draws
        shifted_draws = draws - draws.iloc[min_index]
        # Lower bound for draws across daily mean temperature
        draws_lb = np.quantile(shifted_draws, 0.05, axis=1)
        # Return the score by estimating the area between lower bound and x-axis
        score = np.mean(draws_lb)
        score_dict[annual_temp] = np.round(score, 4)
    df_score = pd.DataFrame(list(score_dict.items()),columns = ['annual_temperature', 'score'])    
    return df_score

