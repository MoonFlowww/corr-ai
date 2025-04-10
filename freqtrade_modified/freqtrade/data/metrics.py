import logging
import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def calculate_market_change(data: dict[str, pd.DataFrame], column: str = "close") -> float:
    """
    Calculate market change based on "column".
    Calculation is done by taking the first non-null and the last non-null element of each column
    and calculating the pctchange as "(last - first) / first".
    Then the results per pair are combined as mean.

    :param data: Dict of Dataframes, dict key should be pair.
    :param column: Column in the original dataframes to use
    :return:
    """
    tmp_means = []
    for pair, df in data.items():
        start = df[column].dropna().iloc[0]
        end = df[column].dropna().iloc[-1]
        tmp_means.append((end - start) / start)

    return float(np.mean(tmp_means))


def combine_dataframes_by_column(
    data: dict[str, pd.DataFrame], column: str = "close"
) -> pd.DataFrame:
    """
    Combine multiple dataframes "column"
    :param data: Dict of Dataframes, dict key should be pair.
    :param column: Column in the original dataframes to use
    :return: DataFrame with the column renamed to the dict key.
    :raise: ValueError if no data is provided.
    """
    if not data:
        raise ValueError("No data provided.")
    df_comb = pd.concat(
        [data[pair].set_index("date").rename({column: pair}, axis=1)[pair] for pair in data], axis=1
    )
    return df_comb


def combined_dataframes_with_rel_mean(
    data: dict[str, pd.DataFrame], fromdt: datetime, todt: datetime, column: str = "close"
) -> pd.DataFrame:
    """
    Combine multiple dataframes "column"
    :param data: Dict of Dataframes, dict key should be pair.
    :param column: Column in the original dataframes to use
    :return: DataFrame with the column renamed to the dict key, and a column
        named mean, containing the mean of all pairs.
    :raise: ValueError if no data is provided.
    """
    df_comb = combine_dataframes_by_column(data, column)
    # Trim dataframes to the given timeframe
    df_comb = df_comb.iloc[(df_comb.index >= fromdt) & (df_comb.index < todt)]
    df_comb["count"] = df_comb.count(axis=1)
    df_comb["mean"] = df_comb.mean(axis=1)
    df_comb["rel_mean"] = df_comb["mean"].pct_change().fillna(0).cumsum()
    return df_comb[["mean", "rel_mean", "count"]]


def combine_dataframes_with_mean(
    data: dict[str, pd.DataFrame], column: str = "close"
) -> pd.DataFrame:
    """
    Combine multiple dataframes "column"
    :param data: Dict of Dataframes, dict key should be pair.
    :param column: Column in the original dataframes to use
    :return: DataFrame with the column renamed to the dict key, and a column
        named mean, containing the mean of all pairs.
    :raise: ValueError if no data is provided.
    """
    df_comb = combine_dataframes_by_column(data, column)

    df_comb["mean"] = df_comb.mean(axis=1)

    return df_comb


def create_cum_profit(
    df: pd.DataFrame, trades: pd.DataFrame, col_name: str, timeframe: str
) -> pd.DataFrame:
    """
    Adds a column `col_name` with the cumulative profit for the given trades array.
    :param df: DataFrame with date index
    :param trades: DataFrame containing trades (requires columns close_date and profit_abs)
    :param col_name: Column name that will be assigned the results
    :param timeframe: Timeframe used during the operations
    :return: Returns df with one additional column, col_name, containing the cumulative profit.
    :raise: ValueError if trade-dataframe was found empty.
    """
    if len(trades) == 0:
        raise ValueError("Trade dataframe empty.")
    from freqtrade.exchange import timeframe_to_resample_freq

    timeframe_freq = timeframe_to_resample_freq(timeframe)
    # Resample to timeframe to make sure trades match candles
    _trades_sum = trades.resample(timeframe_freq, on="close_date")[["profit_abs"]].sum()
    df.loc[:, col_name] = _trades_sum["profit_abs"].cumsum()
    # Set first value to 0
    df.loc[df.iloc[0].name, col_name] = 0
    # FFill to get continuous
    df[col_name] = df[col_name].ffill()
    return df


def _calc_drawdown_series(
    profit_results: pd.DataFrame, *, date_col: str, value_col: str, starting_balance: float
) -> pd.DataFrame:
    max_drawdown_df = pd.DataFrame()
    max_drawdown_df["cumulative"] = profit_results[value_col].cumsum()
    max_drawdown_df["high_value"] = max_drawdown_df["cumulative"].cummax()
    max_drawdown_df["drawdown"] = max_drawdown_df["cumulative"] - max_drawdown_df["high_value"]
    max_drawdown_df["date"] = profit_results.loc[:, date_col]
    if starting_balance:
        cumulative_balance = starting_balance + max_drawdown_df["cumulative"]
        max_balance = starting_balance + max_drawdown_df["high_value"]
        max_drawdown_df["drawdown_relative"] = (max_balance - cumulative_balance) / max_balance
    else:
        # NOTE: This is not completely accurate,
        # but might good enough if starting_balance is not available
        max_drawdown_df["drawdown_relative"] = (
            max_drawdown_df["high_value"] - max_drawdown_df["cumulative"]
        ) / max_drawdown_df["high_value"]
    return max_drawdown_df


def calculate_underwater(
    trades: pd.DataFrame,
    *,
    date_col: str = "close_date",
    value_col: str = "profit_ratio",
    starting_balance: float = 0.0,
):
    """
    Calculate max drawdown and the corresponding close dates
    :param trades: DataFrame containing trades (requires columns close_date and profit_ratio)
    :param date_col: Column in DataFrame to use for dates (defaults to 'close_date')
    :param value_col: Column in DataFrame to use for values (defaults to 'profit_ratio')
    :return: Tuple (float, highdate, lowdate, highvalue, lowvalue) with absolute max drawdown,
             high and low time and high and low value.
    :raise: ValueError if trade-dataframe was found empty.
    """
    if len(trades) == 0:
        raise ValueError("Trade dataframe empty.")
    profit_results = trades.sort_values(date_col).reset_index(drop=True)
    max_drawdown_df = _calc_drawdown_series(
        profit_results, date_col=date_col, value_col=value_col, starting_balance=starting_balance
    )

    return max_drawdown_df


@dataclass()
class DrawDownResult:
    drawdown_abs: float = 0.0
    high_date: pd.Timestamp = None
    low_date: pd.Timestamp = None
    high_value: float = 0.0
    low_value: float = 0.0
    relative_account_drawdown: float = 0.0


def calculate_max_drawdown(
    trades: pd.DataFrame,
    *,
    date_col: str = "close_date",
    value_col: str = "profit_abs",
    starting_balance: float = 0,
    relative: bool = False,
) -> DrawDownResult:
    """
    Calculate max drawdown and the corresponding close dates
    :param trades: DataFrame containing trades (requires columns close_date and profit_ratio)
    :param date_col: Column in DataFrame to use for dates (defaults to 'close_date')
    :param value_col: Column in DataFrame to use for values (defaults to 'profit_abs')
    :param starting_balance: Portfolio starting balance - properly calculate relative drawdown.
    :return: DrawDownResult object
             with absolute max drawdown, high and low time and high and low value,
             and the relative account drawdown
    :raise: ValueError if trade-dataframe was found empty.
    """
    if len(trades) == 0:
        raise ValueError("Trade dataframe empty.")
    profit_results = trades.sort_values(date_col).reset_index(drop=True)
    max_drawdown_df = _calc_drawdown_series(
        profit_results, date_col=date_col, value_col=value_col, starting_balance=starting_balance
    )

    idxmin = (
        max_drawdown_df["drawdown_relative"].idxmax()
        if relative
        else max_drawdown_df["drawdown"].idxmin()
    )
    if idxmin == 0:
        raise ValueError("No losing trade, therefore no drawdown.")
    high_date = profit_results.loc[max_drawdown_df.iloc[:idxmin]["high_value"].idxmax(), date_col]
    low_date = profit_results.loc[idxmin, date_col]
    high_val = max_drawdown_df.loc[
        max_drawdown_df.iloc[:idxmin]["high_value"].idxmax(), "cumulative"
    ]
    low_val = max_drawdown_df.loc[idxmin, "cumulative"]
    max_drawdown_rel = max_drawdown_df.loc[idxmin, "drawdown_relative"]

    return DrawDownResult(
        drawdown_abs=abs(max_drawdown_df.loc[idxmin, "drawdown"]),
        high_date=high_date,
        low_date=low_date,
        high_value=high_val,
        low_value=low_val,
        relative_account_drawdown=max_drawdown_rel,
    )


def calculate_csum(trades: pd.DataFrame, starting_balance: float = 0) -> tuple[float, float]:
    """
    Calculate min/max cumsum of trades, to show if the wallet/stake amount ratio is sane
    :param trades: DataFrame containing trades (requires columns close_date and profit_percent)
    :param starting_balance: Add starting balance to results, to show the wallets high / low points
    :return: Tuple (float, float) with cumsum of profit_abs
    :raise: ValueError if trade-dataframe was found empty.
    """
    if len(trades) == 0:
        raise ValueError("Trade dataframe empty.")

    csum_df = pd.DataFrame()
    csum_df["sum"] = trades["profit_abs"].cumsum()
    csum_min = csum_df["sum"].min() + starting_balance
    csum_max = csum_df["sum"].max() + starting_balance

    return csum_min, csum_max


def calculate_cagr(days_passed: int, starting_balance: float, final_balance: float) -> float:
    """
    Calculate CAGR
    :param days_passed: Days passed between start and ending balance
    :param starting_balance: Starting balance
    :param final_balance: Final balance to calculate CAGR against
    :return: CAGR
    """
    if final_balance < 0:
        # With leveraged trades, final_balance can become negative.
        return 0
    return (final_balance / starting_balance) ** (1 / (days_passed / 365)) - 1


def calculate_expectancy(trades: pd.DataFrame) -> tuple[float, float]:
    """
    Calculate expectancy
    :param trades: DataFrame containing trades (requires columns close_date and profit_abs)
    :return: expectancy, expectancy_ratio
    """

    expectancy = 0.0
    expectancy_ratio = 100.0

    if len(trades) > 0:
        winning_trades = trades.loc[trades["profit_abs"] > 0]
        losing_trades = trades.loc[trades["profit_abs"] < 0]
        profit_sum = winning_trades["profit_abs"].sum()
        loss_sum = abs(losing_trades["profit_abs"].sum())
        nb_win_trades = len(winning_trades)
        nb_loss_trades = len(losing_trades)

        average_win = (profit_sum / nb_win_trades) if nb_win_trades > 0 else 0
        average_loss = (loss_sum / nb_loss_trades) if nb_loss_trades > 0 else 0
        winrate = nb_win_trades / len(trades)
        loserate = nb_loss_trades / len(trades)

        expectancy = (winrate * average_win) - (loserate * average_loss)
        if average_loss > 0:
            risk_reward_ratio = average_win / average_loss
            expectancy_ratio = ((1 + risk_reward_ratio) * winrate) - 1

    return expectancy, expectancy_ratio


def calculate_sortino(
    trades: pd.DataFrame, min_date: datetime, max_date: datetime, starting_balance: float
) -> float:
    """
    Calculate sortino
    :param trades: DataFrame containing trades (requires columns profit_abs)
    :return: sortino
    """
    if (len(trades) == 0) or (min_date is None) or (max_date is None) or (min_date == max_date):
        return 0

    total_profit = trades["profit_abs"] / starting_balance
    days_period = max(1, (max_date - min_date).days)

    expected_returns_mean = total_profit.sum() / days_period

    down_stdev = np.std(trades.loc[trades["profit_abs"] < 0, "profit_abs"] / starting_balance)

    if down_stdev != 0 and not np.isnan(down_stdev):
        sortino_ratio = expected_returns_mean / down_stdev * np.sqrt(365)
    else:
        # Define high (negative) sortino ratio to be clear that this is NOT optimal.
        sortino_ratio = -100

    # print(expected_returns_mean, down_stdev, sortino_ratio)
    return sortino_ratio


def calculate_sharpe(
    trades: pd.DataFrame, min_date: datetime, max_date: datetime, starting_balance: float
) -> float:
    """
    Calculate sharpe
    :param trades: DataFrame containing trades (requires column profit_abs)
    :return: sharpe
    """
    if (len(trades) == 0) or (min_date is None) or (max_date is None) or (min_date == max_date):
        return 0

    total_profit = trades["profit_abs"] / starting_balance
    days_period = max(1, (max_date - min_date).days)

    expected_returns_mean = total_profit.sum() / days_period
    up_stdev = np.std(total_profit)

    if up_stdev != 0:
        sharp_ratio = expected_returns_mean / up_stdev * np.sqrt(365)
    else:
        # Define high (negative) sharpe ratio to be clear that this is NOT optimal.
        sharp_ratio = -100

    # print(expected_returns_mean, up_stdev, sharp_ratio)
    return sharp_ratio


def calculate_calmar(
    trades: pd.DataFrame, min_date: datetime, max_date: datetime, starting_balance: float
) -> float:
    """
    Calculate calmar
    :param trades: DataFrame containing trades (requires columns close_date and profit_abs)
    :return: calmar
    """
    if (len(trades) == 0) or (min_date is None) or (max_date is None) or (min_date == max_date):
        return 0

    total_profit = trades["profit_abs"].sum() / starting_balance
    days_period = max(1, (max_date - min_date).days)

    # adding slippage of 0.1% per trade
    # total_profit = total_profit - 0.0005
    expected_returns_mean = total_profit / days_period * 100

    # calculate max drawdown
    try:
        drawdown = calculate_max_drawdown(
            trades, value_col="profit_abs", starting_balance=starting_balance
        )
        max_drawdown = drawdown.relative_account_drawdown
    except ValueError:
        max_drawdown = 0

    if max_drawdown != 0:
        calmar_ratio = expected_returns_mean / max_drawdown * math.sqrt(365)
    else:
        # Define high (negative) calmar ratio to be clear that this is NOT optimal.
        calmar_ratio = -100

    # print(expected_returns_mean, max_drawdown, calmar_ratio)
    return calmar_ratio


def calculate_ulcer_index(trades: pd.DataFrame, starting_balance: float) -> float:
    """
    Calculate the Ulcer Index (UI)
    :param trades: DataFrame containing trades (requires columns close_date and profit_abs)
    :param starting_balance: Starting balance for calculations
    :return: Ulcer Index value
    """
    if len(trades) == 0:
        return 0.0

    underwater_df = calculate_underwater(
        trades=trades,
        date_col='close_date',
        value_col='profit_abs',
        starting_balance=starting_balance
    )
    #Mean absolute deviation (MAD)
    squared_drawdowns = (underwater_df['drawdown_relative'] ** 2)
    mean_squared_drawdown = squared_drawdowns.mean()
    ulcer_index = math.sqrt(mean_squared_drawdown)

    return ulcer_index


def calculate_ulcer_performance_index(
    trades: pd.DataFrame,
    min_date: datetime,
    max_date: datetime,
    starting_balance: float,
    final_balance: float
) -> float:
    """
    Calculate the Ulcer Performance Index (UPI)
    :param trades: DataFrame containing trades (requires columns close_date and profit_abs)
    :param min_date: Start date of the period
    :param max_date: End date of the period
    :param starting_balance: Starting balance
    :param final_balance: Final balance
    :return: Ulcer Performance Index value
    """
    if (len(trades) == 0) or (min_date is None) or (max_date is None) or (min_date == max_date):
        return 0.0

    days_passed = (max_date - min_date).days
    cagr = calculate_cagr(days_passed, starting_balance, final_balance)

    ulcer_index = calculate_ulcer_index(trades, starting_balance)

    if ulcer_index != 0:
        upi = cagr / ulcer_index
    else: #same logic as Sharpe/Calmar
        upi = -100.0

    return upi


def calculate_information_ratio(
    trades: pd.DataFrame,
    market_data: dict[str, pd.DataFrame], #TODO: check how to find benchmark data
    min_date: datetime,
    max_date: datetime,
    starting_balance: float
) -> float:
    """
    Calculate the Information Ratio (IR)
    :param trades: DataFrame containing trades (requires columns close_date and profit_abs)
    :param market_data: Dict of Dataframes containing market data for benchmark
    :param min_date: Start date of the period
    :param max_date: End date of the period
    :param starting_balance: Starting balance
    :return: Information Ratio value
    """
    if (len(trades) == 0) or (min_date is None) or (max_date is None) or (min_date == max_date):
        return 0.0

    total_profit = trades["profit_abs"] / starting_balance
    days_period = max(1, (max_date - min_date).days)
    avg_returns = total_profit.sum() / days_period

    benchmark_returns = calculate_market_change(market_data, column="close")

    market_returns_series = combined_dataframes_with_rel_mean(
        market_data, min_date, max_date, column="close"
    )["rel_mean"].pct_change().fillna(0)

    excess_returns = total_profit - market_returns_series

    tracking_error = np.std(excess_returns)

    # IR
    if tracking_error != 0:
        information_ratio = (avg_returns - benchmark_returns) / tracking_error
    else:
        information_ratio = -100.0

    return information_ratio


def calculate_var(
    trades: pd.DataFrame,
    starting_balance: float,
    quantile: float = 0.05
) -> float:
    """
    Calculate Value at Risk (VaR) at specified quantile
    :param trades: DataFrame containing trades (requires column profit_abs)
    :param starting_balance: Starting balance for calculations
    :param quantile: Quantile level for VaR calculation (default 0.05 for 95% confidence)
    :return: VaR value (negative value represents loss)
    """
    if len(trades) == 0:
        return 0.0

    returns = trades["profit_abs"] / starting_balance

    var = np.quantile(returns, quantile)

    return var


def calculate_cvar(
    trades: pd.DataFrame,
    starting_balance: float,
    quantile: float = 0.05
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) (risk measure)
    :param trades: DataFrame containing trades (requires column profit_abs)
    :param starting_balance: Starting balance for calculations
    :param quantile: Quantile level for CVaR calculation (default 0.05 for 95% confidence)
    :return: CVaR value (negative value represents loss)
    """
    if len(trades) == 0:
        return 0.0

    returns = trades["profit_abs"] / starting_balance

    var = calculate_var(trades, starting_balance, quantile)

    cvar = returns[returns <= var].mean()

    return cvar

def calculate_rachev(
    trades: pd.DataFrame,
    starting_balance: float,
    quantile: float = 0.05
) -> float:
    """
    Calculate Rachev Ratio (risk measure)
    :param trades: DataFrame containing trades (requires column profit_abs)
    :param starting_balance: Starting balance for calculations
    :param quantile: Quantile level for calculation (default 0.05 for 95% confidence)
    :return: Rachev Ratio value (higher is better)
    """
    if len(trades) == 0:
        return 0.0

    returns = trades["profit_abs"] / starting_balance

    # negative kurtosis
    var_lower = np.quantile(returns, quantile)
    cvar_lower = returns[returns <= var_lower].mean()

    #Positive kustosis
    var_upper = np.quantile(returns, 1 - quantile)
    cvar_upper = returns[returns >= var_upper].mean()

    if cvar_upper != 0 and cvar_lower != 0:
        rachev_ratio = abs(cvar_upper / cvar_lower)
    else:
        rachev_ratio = -100.0

    return rachev_ratio

def calculate_recovery_ratio(
    trades: pd.DataFrame,
    starting_balance: float
) -> float:
    """
    Calculate Recovery Ratio (risk measure) how many trades do we need to recover from the max drawdown
    Only give result if avg_return is positive (profitable edge)
    :param trades: DataFrame containing trades (requires columns close_date and profit_abs)
    :param starting_balance: Starting balance for calculations
    :return: Recovery Ratio value (lower is better)
    """
    if len(trades) == 0:
        return 0.0

    try:
        drawdown = calculate_max_drawdown(
            trades, value_col="profit_abs", starting_balance=starting_balance
        )
        mdd = drawdown.relative_account_drawdown
    except ValueError:
        mdd = 0.0

    avg_return = trades["profit_abs"].mean() / starting_balance

    if avg_return > 0: #only work with positive avg_return
        recovery_ratio = mdd / avg_return
    else:
        recovery_ratio = -100.0

    return recovery_ratio


def calculate_kelly_criterion(
    trades: pd.DataFrame,
    starting_balance: float
) -> float:
    """
    Calculate the Kelly Criterion to suggest optimal position sizing.
    It determines the fraction of capital to allocate to each trade.
    Formula: K% = W - [(1 - W) / R] where W is win rate and R is win/loss ratio.
    :param trades: DataFrame containing trades (requires column profit_abs)
    :param starting_balance: Starting balance for calculations
    :return: Kelly Criterion ratio (suggested fraction of capital per trade). Returns -100.0 if calculation is not possible (e.g., no losses).
    """
    if len(trades) == 0:
        return 0.0

    try:
        winning_trades = trades[trades["profit_abs"] > 0]
        avg_positive_return = (winning_trades["profit_abs"].mean() / starting_balance) if not winning_trades.empty else 0.0

        losing_trades = trades[trades["profit_abs"] < 0]
        avg_negative_return = (losing_trades["profit_abs"].mean() / starting_balance) if not losing_trades.empty else 0.0
        win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0.0
    except ValueError:
        mdd = 0.0

    

    if len(winning_trades)>0  and len(losing_trades)>0:
        kelly_ratio = win_rate * ((1-win_rate)/abs(avg_positive_return/avg_negative_return)) #P * (1-P)/R
    else:
        kelly_ratio = -100.0

    return kelly_ratio
