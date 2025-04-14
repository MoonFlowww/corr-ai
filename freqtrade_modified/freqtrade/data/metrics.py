import logging
import math
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Dict, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

_metrics_cache = {}


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
    changes = np.array([
        (df[column].dropna().iloc[-1] - df[column].dropna().iloc[0]) / df[column].dropna().iloc[0]
        for pair, df in data.items()
    ])
    
    return float(np.mean(changes))


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
    
    cache_key = f"combine_df_{column}_{'-'.join(sorted(data.keys()))}"
    if cache_key in _metrics_cache:
        return _metrics_cache[cache_key]
    
    dfs_to_concat = []
    for pair, df in data.items():
        df_renamed = df.set_index("date").rename({column: pair}, axis=1)[pair]
        dfs_to_concat.append(df_renamed)
    
    df_comb = pd.concat(dfs_to_concat, axis=1)
    
    _metrics_cache[cache_key] = df_comb
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
    cache_key = f"combined_rel_mean_{column}_{fromdt}_{todt}_{'-'.join(sorted(data.keys()))}"
    if cache_key in _metrics_cache:
        return _metrics_cache[cache_key]
    
    df_comb = combine_dataframes_by_column(data, column)
    mask = (df_comb.index >= fromdt) & (df_comb.index < todt)
    df_comb = df_comb.loc[mask]
    
    df_comb = df_comb.assign(
        count=lambda x: x.count(axis=1),
        mean=lambda x: x.mean(axis=1)
    )
    df_comb["rel_mean"] = df_comb["mean"].pct_change().fillna(0).cumsum()
    
    result = df_comb[["mean", "rel_mean", "count"]]
    _metrics_cache[cache_key] = result
    return result


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
    cache_key = f"combined_mean_{column}_{'-'.join(sorted(data.keys()))}"
    if cache_key in _metrics_cache:
        return _metrics_cache[cache_key]
    
    df_comb = combine_dataframes_by_column(data, column)
    df_comb["mean"] = df_comb.mean(axis=1)
    
    _metrics_cache[cache_key] = df_comb
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
    
    cache_key = f"cum_profit_{col_name}_{timeframe}_{id(df)}_{id(trades)}"
    if cache_key in _metrics_cache:
        return _metrics_cache[cache_key]
    
    from freqtrade.exchange import timeframe_to_resample_freq

    timeframe_freq = timeframe_to_resample_freq(timeframe)
    _trades_sum = trades.resample(timeframe_freq, on="close_date")[["profit_abs"]].sum()
    
    df = df.copy()  # keep the input safe
    df.loc[:, col_name] = 0  # memory efficient
    df.loc[:, col_name] = _trades_sum["profit_abs"].cumsum()
    df.loc[df.iloc[0].name, col_name] = 0
    df[col_name] = df[col_name].ffill()
    
    _metrics_cache[cache_key] = df
    return df


def _calc_drawdown_series(
    profit_results: pd.DataFrame, *, date_col: str, value_col: str, starting_balance: float
) -> pd.DataFrame:
    """Calculate drawdown series with vectorized operations"""
    cumulative = profit_results[value_col].cumsum()
    high_value = cumulative.cummax()
    drawdown = cumulative - high_value
    
    max_drawdown_df = pd.DataFrame({
        "cumulative": cumulative,
        "high_value": high_value,
        "drawdown": drawdown,
        "date": profit_results[date_col].values
    })
    
    if starting_balance:
        cumulative_balance = starting_balance + max_drawdown_df["cumulative"]
        max_balance = starting_balance + max_drawdown_df["high_value"]
        max_drawdown_df["drawdown_relative"] = (max_balance - cumulative_balance) / max_balance
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            max_drawdown_df["drawdown_relative"] = (
                (max_drawdown_df["high_value"] - max_drawdown_df["cumulative"]) / 
                max_drawdown_df["high_value"]
            )
        max_drawdown_df["drawdown_relative"].replace([np.inf, -np.inf], np.nan, inplace=True)
        max_drawdown_df["drawdown_relative"].fillna(0, inplace=True)
        
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
    :return: DataFrame with drawdown metrics
    :raise: ValueError if trade-dataframe was found empty.
    """
    if len(trades) == 0:
        raise ValueError("Trade dataframe empty.")
    
    cache_key = f"underwater_{id(trades)}_{date_col}_{value_col}_{starting_balance}"
    if cache_key in _metrics_cache:
        return _metrics_cache[cache_key]
    
    profit_results = trades.sort_values(date_col).reset_index(drop=True)
    max_drawdown_df = _calc_drawdown_series(
        profit_results, date_col=date_col, value_col=value_col, starting_balance=starting_balance
    )
    
    _metrics_cache[cache_key] = max_drawdown_df
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
    
    cache_key = f"max_drawdown_{id(trades)}_{date_col}_{value_col}_{starting_balance}_{relative}"
    if cache_key in _metrics_cache:
        return _metrics_cache[cache_key]
    
    underwater_key = f"underwater_{id(trades)}_{date_col}_{value_col}_{starting_balance}"
    if underwater_key in _metrics_cache:
        max_drawdown_df = _metrics_cache[underwater_key]
    else:
        profit_results = trades.sort_values(date_col).reset_index(drop=True)
        max_drawdown_df = _calc_drawdown_series(
            profit_results, date_col=date_col, value_col=value_col, starting_balance=starting_balance
        )
        _metrics_cache[underwater_key] = max_drawdown_df
    
    idxmin = (
        max_drawdown_df["drawdown_relative"].idxmax()
        if relative
        else max_drawdown_df["drawdown"].idxmin()
    )
    
    if idxmin == 0:
        raise ValueError("No losing trade, therefore no drawdown.")
    
    profit_results = trades.sort_values(date_col).reset_index(drop=True)
    
    high_idx = max_drawdown_df.iloc[:idxmin]["high_value"].idxmax()
    high_date = profit_results.loc[high_idx, date_col]
    low_date = profit_results.loc[idxmin, date_col]
    high_val = max_drawdown_df.loc[high_idx, "cumulative"]
    low_val = max_drawdown_df.loc[idxmin, "cumulative"]
    max_drawdown_rel = max_drawdown_df.loc[idxmin, "drawdown_relative"]

    result = DrawDownResult(
        drawdown_abs=abs(max_drawdown_df.loc[idxmin, "drawdown"]),
        high_date=high_date,
        low_date=low_date,
        high_value=high_val,
        low_value=low_val,
        relative_account_drawdown=max_drawdown_rel,
    )
    
    _metrics_cache[cache_key] = result
    return result


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
    
    cache_key = f"csum_{id(trades)}_{starting_balance}"
    if cache_key in _metrics_cache:
        return _metrics_cache[cache_key]
    
    cumsum = trades["profit_abs"].cumsum()
    csum_min = cumsum.min() + starting_balance
    csum_max = cumsum.max() + starting_balance
    
    result = (csum_min, csum_max)
    _metrics_cache[cache_key] = result
    return result


def calculate_cagr(days_passed: int, starting_balance: float, final_balance: float) -> float:
    """
    Calculate CAGR
    :param days_passed: Days passed between start and ending balance
    :param starting_balance: Starting balance
    :param final_balance: Final balance to calculate CAGR against
    :return: CAGR
    """
    if final_balance < 0:
        return 0
    
    return (final_balance / starting_balance) ** (1 / (days_passed / 365)) - 1


def calculate_expectancy(trades: pd.DataFrame) -> tuple[float, float]:
    """
    Calculate expectancy
    :param trades: DataFrame containing trades (requires columns close_date and profit_abs)
    :return: expectancy, expectancy_ratio
    """
    cache_key = f"expectancy_{id(trades)}"
    if cache_key in _metrics_cache:
        return _metrics_cache[cache_key]
    
    expectancy = 0.0
    expectancy_ratio = 100.0

    if len(trades) > 0:
        is_win = trades["profit_abs"] > 0
        
        winning_trades = trades[is_win]
        losing_trades = trades[~is_win & (trades["profit_abs"] < 0)] 
        
        nb_win_trades = len(winning_trades)
        nb_loss_trades = len(losing_trades)
        total_trades = len(trades)
        
        # sums and means
        profit_sum = winning_trades["profit_abs"].sum() if nb_win_trades > 0 else 0
        loss_sum = abs(losing_trades["profit_abs"].sum()) if nb_loss_trades > 0 else 0
        average_win = profit_sum / nb_win_trades if nb_win_trades > 0 else 0
        average_loss = loss_sum / nb_loss_trades if nb_loss_trades > 0 else 0
        
        # rates
        winrate = nb_win_trades / total_trades if total_trades > 0 else 0
        loserate = nb_loss_trades / total_trades if total_trades > 0 else 0

        expectancy = (winrate * average_win) - (loserate * average_loss)
        if average_loss > 0:
            risk_reward_ratio = average_win / average_loss
            expectancy_ratio = ((1 + risk_reward_ratio) * winrate) - 1
    
    result = (expectancy, expectancy_ratio)
    _metrics_cache[cache_key] = result
    return result


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
    
    cache_key = f"sortino_{id(trades)}_{min_date}_{max_date}_{starting_balance}"
    if cache_key in _metrics_cache:
        return _metrics_cache[cache_key]

    # normalize profit_$ -> profit_%_equity
    profit_ratio = trades["profit_abs"] / starting_balance
    
    days_period = max(1, (max_date - min_date).days)
    
    expected_returns_mean = profit_ratio.sum() / days_period
    
    losing_mask = trades["profit_abs"] < 0
    losing_profits = profit_ratio[losing_mask]
    down_stdev = np.std(losing_profits) if len(losing_profits) > 0 else 0

    if down_stdev != 0 and not np.isnan(down_stdev):
        sortino_ratio = expected_returns_mean / down_stdev * np.sqrt(365)
    else:
        # does it cause trouble if we set nan ?
        sortino_ratio = -100
    
    _metrics_cache[cache_key] = sortino_ratio
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
    
    cache_key = f"sharpe_{id(trades)}_{min_date}_{max_date}_{starting_balance}"
    if cache_key in _metrics_cache:
        return _metrics_cache[cache_key]

    profit_ratio = trades["profit_abs"] / starting_balance
    
    days_period = max(1, (max_date - min_date).days)
    
    expected_returns_mean = profit_ratio.sum() / days_period
    up_stdev = np.std(profit_ratio)

    if up_stdev != 0:
        sharp_ratio = expected_returns_mean / up_stdev * np.sqrt(365)
    else:
        sharp_ratio = -100
    
    _metrics_cache[cache_key] = sharp_ratio
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
    
    cache_key = f"calmar_{id(trades)}_{min_date}_{max_date}_{starting_balance}"
    if cache_key in _metrics_cache:
        return _metrics_cache[cache_key]

    total_profit = trades["profit_abs"].sum() / starting_balance
    
    days_period = max(1, (max_date - min_date).days)
    
    expected_returns_mean = total_profit / days_period * 100

    try:
        dd_key = f"max_drawdown_{id(trades)}_close_date_profit_abs_{starting_balance}_False"
        
        if dd_key in _metrics_cache:
            drawdown = _metrics_cache[dd_key]
        else:
            drawdown = calculate_max_drawdown(
                trades, value_col="profit_abs", starting_balance=starting_balance
            )
            _metrics_cache[dd_key] = drawdown
            
        max_drawdown = drawdown.relative_account_drawdown
    except ValueError:
        max_drawdown = 0

    if max_drawdown != 0:
        calmar_ratio = expected_returns_mean / max_drawdown * math.sqrt(365)
    else:
        calmar_ratio = -100
    
    _metrics_cache[cache_key] = calmar_ratio
    return calmar_ratio



def calculate_ulcer_index(
    trades: pd.DataFrame, starting_balance: float = 0
) -> float:
    """
    Calculate Ulcer Index - measures downside risk
    :param trades: DataFrame containing trades
    :param starting_balance: Starting balance amount
    :return: Ulcer Index value
    """
    if len(trades) == 0:
        return 0
    
    cache_key = f"ulcer_index_{id(trades)}_{starting_balance}"
    if cache_key in _metrics_cache:
        return _metrics_cache[cache_key]
    
    underwater_key = f"underwater_{id(trades)}_close_date_profit_abs_{starting_balance}"
    if underwater_key in _metrics_cache:
        drawdown_df = _metrics_cache[underwater_key]
    else:
        drawdown_df = calculate_underwater(
            trades, date_col="close_date", value_col="profit_abs", starting_balance=starting_balance
        )
    
    squared_drawdowns = np.square(drawdown_df["drawdown_relative"] * 100)
    
    ulcer_index = np.sqrt(squared_drawdowns.mean())
    
    _metrics_cache[cache_key] = ulcer_index
    return ulcer_index


def calculate_ulcer_performance_index(
    trades: pd.DataFrame, min_date: datetime, max_date: datetime, 
    starting_balance: float, final_balance: float
) -> float:
    """
    Calculate Ulcer Performance Index (UPI)
    :param trades: DataFrame containing trades
    :param min_date: Start date of trading period
    :param max_date: End date of trading period
    :param starting_balance: Starting balance
    :param final_balance: Final balance after trading
    :return: UPI value
    """
    if (len(trades) == 0) or (min_date is None) or (max_date is None) or (min_date == max_date):
        return 0
    
    cache_key = f"upi_{id(trades)}_{min_date}_{max_date}_{starting_balance}_{final_balance}"
    if cache_key in _metrics_cache:
        return _metrics_cache[cache_key]
    
    days_passed = max(1, (max_date - min_date).days)
    annual_return = ((final_balance / starting_balance) ** (365 / days_passed) - 1) * 100
    
    ulcer_index = calculate_ulcer_index(trades, starting_balance)
    
    if ulcer_index > 0:
        upi = annual_return / ulcer_index
    else:
        upi = 0
    
    _metrics_cache[cache_key] = upi
    return upi


def calculate_var(
    trades: pd.DataFrame, starting_balance: float, confidence: float = 0.95
) -> float:
    """
    Calculate Value at Risk (VaR)
    :param trades: DataFrame containing trades
    :param starting_balance: Starting balance
    :param confidence: Confidence level (default 95%)
    :return: VaR value
    """
    if len(trades) == 0:
        return 0
    
    cache_key = f"var_{id(trades)}_{starting_balance}_{confidence}"
    if cache_key in _metrics_cache:
        return _metrics_cache[cache_key]
    
    # norm
    returns = trades["profit_abs"] / starting_balance
    
    var = np.percentile(returns, 100 * (1 - confidence))
    
    _metrics_cache[cache_key] = var
    return var


def calculate_cvar(
    trades: pd.DataFrame, starting_balance: float, confidence: float = 0.95
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) also known as Expected Shortfall
    :param trades: DataFrame containing trades
    :param starting_balance: Starting balance
    :param confidence: Confidence level (default 95%)
    :return: CVaR value
    """
    if len(trades) == 0:
        return 0
    
    cache_key = f"cvar_{id(trades)}_{starting_balance}_{confidence}"
    if cache_key in _metrics_cache:
        return _metrics_cache[cache_key]
    
    returns = trades["profit_abs"] / starting_balance
    
    var = np.percentile(returns, 100 * (1 - confidence))
    
    cvar = returns[returns <= var].mean()
    
    _metrics_cache[cache_key] = cvar
    return cvar


def calculate_rachev(
    trades: pd.DataFrame, starting_balance: float, 
    tail_pct: float = 0.05
) -> float:
    """
    Calculate Rachev ratio (modified version)
    :param trades: DataFrame containing trades
    :param starting_balance: Starting balance
    :param tail_pct: Tail percentile to consider (default 5%)
    :return: Rachev ratio
    """
    if len(trades) == 0:
        return 0
    
    cache_key = f"rachev_{id(trades)}_{starting_balance}_{tail_pct}"
    if cache_key in _metrics_cache:
        return _metrics_cache[cache_key]
    
    returns = trades["profit_abs"] / starting_balance
    
    upper_tail = np.percentile(returns, 100 * (1 - tail_pct))
    lower_tail = np.percentile(returns, 100 * tail_pct)
    
    upper_tail_mean = returns[returns >= upper_tail].mean()
    lower_tail_mean = abs(returns[returns <= lower_tail].mean())
    
    if lower_tail_mean != 0:
        rachev = upper_tail_mean / lower_tail_mean
    else:
        rachev = 0
    
    _metrics_cache[cache_key] = rachev
    return rachev


def calculate_recovery_ratio(
    trades: pd.DataFrame, starting_balance: float
) -> float:
    """
    Calculate Recovery Ratio - measures how efficiently the strategy recovers from drawdowns
    :param trades: DataFrame containing trades
    :param starting_balance: Starting balance
    :return: Recovery ratio
    """
    if len(trades) == 0:
        return 0
    
    cache_key = f"recovery_ratio_{id(trades)}_{starting_balance}"
    if cache_key in _metrics_cache:
        return _metrics_cache[cache_key]
    
    try:
        max_dd = calculate_max_drawdown(trades, starting_balance=starting_balance)
        total_profit = trades["profit_abs"].sum()
        
        if max_dd.drawdown_abs > 0:
            recovery_ratio = total_profit / max_dd.drawdown_abs
        else:
            recovery_ratio = -100
            
    except ValueError:
        recovery_ratio = -100
    
    _metrics_cache[cache_key] = recovery_ratio
    return recovery_ratio


def calculate_kelly_criterion(
    trades: pd.DataFrame, starting_balance: float
) -> float:
    """
    Calculate Kelly Criterion - optimal position sizing fraction
    :param trades: DataFrame containing trades
    :param starting_balance: Starting balance
    :return: Kelly criterion value
    """
    if len(trades) == 0:
        return 0
    
    cache_key = f"kelly_{id(trades)}_{starting_balance}"
    if cache_key in _metrics_cache:
        return _metrics_cache[cache_key]
    
    exp_key = f"expectancy_{id(trades)}"
    if exp_key in _metrics_cache:
        expectancy, expectancy_ratio = _metrics_cache[exp_key]
    else:
        expectancy, expectancy_ratio = calculate_expectancy(trades)
    
    win_mask = trades["profit_abs"] > 0
    winrate = win_mask.mean()
    
    if len(trades[win_mask]) > 0:
        avg_win = trades.loc[win_mask, "profit_abs"].mean() / starting_balance
    else:
        avg_win = 0
        
    if len(trades[~win_mask]) > 0:
        avg_loss = abs(trades.loc[~win_mask, "profit_abs"].mean()) / starting_balance
    else:
        avg_loss = 1 
    
    if avg_loss > 0:
        kelly = winrate - ((1 - winrate) / (avg_win / avg_loss))
    else:
        kelly = -100 # not possible due to 0 losses
    
    kelly = max(0, min(1, kelly))
    
    _metrics_cache[cache_key] = kelly
    return kelly


def calculate_information_ratio(
    trades: pd.DataFrame, market_data: dict, min_date: datetime, max_date: datetime, 
    starting_balance: float
) -> float:
    """
    Calculate Information Ratio - excess return per unit of risk relative to benchmark
    :param trades: DataFrame containing trades
    :param market_data: Market data for benchmark comparison
    :param min_date: Start date
    :param max_date: End date
    :param starting_balance: Starting balance
    :return: Information ratio
    """
    if (len(trades) == 0) or (min_date is None) or (max_date is None) or (min_date == max_date):
        return 0
    
    cache_key = f"info_ratio_{id(trades)}_{'-'.join(sorted(market_data.keys()))}_{min_date}_{max_date}_{starting_balance}"
    if cache_key in _metrics_cache:
        return _metrics_cache[cache_key]
    
    total_profit = trades["profit_abs"].sum()
    strategy_return = total_profit / starting_balance
    
    try:
        benchmark_return = calculate_market_change(market_data)
    except Exception:
        benchmark_return = 0
    
    excess_return = strategy_return - benchmark_return
    
    returns = trades["profit_abs"] / starting_balance
    tracking_error = np.std(returns)
    
    if tracking_error > 0:
        information_ratio = excess_return / tracking_error
    else:
        information_ratio = 0
    
    _metrics_cache[cache_key] = information_ratio
    return information_ratio
