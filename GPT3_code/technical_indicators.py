import numpy as np

def moving_average(data, window_size):
    moving_average = []
    for i in range(len(data) - window_size + 1):
        moving_average.append(sum(data[i:i + window_size]) / window_size)
    return moving_average

def bollinger_bands(data, window_size):
    moving_average = moving_average(data, window_size)
    standard_deviation = []
    for i in range(len(data) - window_size + 1):
        standard_deviation.append(np.std(data[i:i + window_size]))
    upper_band = [moving_average[i] + 2 * standard_deviation[i] for i in range(len(moving_average))]
    lower_band = [moving_average[i] - 2 * standard_deviation[i] for i in range(len(moving_average))]
    return moving_average, upper_band, lower_band

def relative_strength_index(data, window_size):
    gains = []
    losses = []
    for i in range(1, len(data)):
        if data[i] > data[i - 1]:
            gains.append(data[i] - data[i - 1])
            losses.append(0)
        else:
            gains.append(0)
            losses.append(data[i - 1] - data[i])
    average_gain = moving_average(gains, window_size)
    average_loss = moving_average(losses, window_size)
    relative_strength = [average_gain[i] / average_loss[i] for i in range(len(average_gain))]
    relative_strength_index = [100 - (100 / (1 + relative_strength[i])) for i in range(len(relative_strength))]
    return relative_strength_index

def stochastic_oscillator(high, low, close, window_size):
    stochastic_oscillator = []
    for i in range(len(high) - window_size + 1):
        highest_high = max(high[i:i + window_size])
        lowest_low = min(low[i:i + window_size])
        stochastic_oscillator.append((close[i + window_size - 1] - lowest_low) / (highest_high - lowest_low))
    return stochastic_oscillator

def directional_movement_index(high, low, close, window_size):
    # Calculate the positive and negative directional movement
    positive_directional_movement = []
    negative_directional_movement = []
    for i in range(1, len(high)):
        positive_directional_movement.append(
            max(high[i] - high[i - 1], 0) if high[i] - high[i - 1] > low[i - 1] - low[i] else 0)
        negative_directional_movement.append(
            max(low[i - 1] - low[i], 0) if high[i] - high[i - 1] < low[i - 1] - low[i] else 0)

    # Calculate the average positive and negative directional movement
    average_positive_directional_movement = moving_average(positive_directional_movement, window_size)
    average_negative_directional_movement = moving_average(negative_directional_movement, window_size)

    # Calculate the directional movement index
    directional_movement_index = []
    for i in range(len(average_positive_directional_movement)):
        directional_movement_index.append(
            100 * average_positive_directional_movement[i] / (average_positive_directional_movement[i] + average_negative_directional_movement[i]))
    return directional_movement_index

def macd(data, n_fast=12, n_slow=26, n_sign=9):
    ema_fast = data.ewm(span=n_fast).mean()
    ema_slow = data.ewm(span=n_slow).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=n_sign).mean()
    hist = macd - signal
    return hist


def dmi(data, n=14):
    data['up'] = data['High'].diff()
    data['down'] = -data['Low'].diff()
    data['up'] = data['up'].where(data['up'] > 0, 0)
    data['down'] = data['down'].where(data['down'] > 0, 0)
    atr = data[['High', 'Low', 'Close']].rolling(n).agg(lambda x: max(x) - min(x))
    up_i = data['up'].rolling(n).mean() / atr['Close']
    down_i = data['down'].rolling(n).mean() / atr['Close']
    dx = 100 * (up_i / (up_i + down_i)).rolling(n).mean().dropna()
    adx = 100 * (dx.ewm(span=n).mean().dropna() / dx.max())
    return adx


def fibonacci_retracement(data, high, low):
    fib = []
    for i in range(len(data)):
        retrace = []
        retrace.append(high[i] - (high[i] - low[i]) * 0.236)
        retrace.append(high[i] - (high[i] - low[i]) * 0.382)
        retrace.append(high[i] - (high[i] - low[i]) * 0.500)
        retrace.append(high[i] - (high[i] - low[i]) * 0.618)
        retrace.append(high[i] - (high[i] - low[i]) * 0.764)
        fib.append(retrace)
    return fib


def aroon_oscillator(df, n):
    aroon_up = df['High'].rolling(window=n).apply(lambda x: np.argmax(x)) / n * 100
    aroon_down = df['Low'].rolling(window=n).apply(lambda x: np.argmin(x)) / n * 100
    aroon_osc = aroon_up - aroon_down
    df['Aroon_Osc'] = aroon_osc
    return df


def aroon_oscillator(df, n):
    aroon_up = df['High'].rolling(window=n).apply(lambda x: np.argmax(x)) / n * 100
    aroon_down = df['Low'].rolling(window=n).apply(lambda x: np.argmin(x)) / n * 100
    aroon_osc = aroon_up - aroon_down
    df['Aroon_Osc'] = aroon_osc
    return df


def on_balance_volume(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'][i] > df['Close'][i-1]:
            obv.append(obv[i-1] + df['Volume'][i])
        elif df['Close'][i] < df['Close'][i-1]:
            obv.append(obv[i-1] - df['Volume'][i])
        else:
            obv.append(obv[i-1])
    df['OBV'] = obv
    return df


def ichimoku_cloud(df, n1, n2):
    tenkan_sen = (df['High'].rolling(window=n1).mean() + df['Low'].rolling(window=n1).mean()) / 2
    kijun_sen = (df['High'].rolling(window=n2).mean() + df['Low'].rolling(window=n2).mean()) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    senkou_span_b = (df['High'].rolling(window=n2).mean() + df['Low'].rolling(window=n2).mean()) / 2
    df['Tenkan_Sen'] = tenkan_sen
    df['Kijun_Sen'] = kijun_sen
    df['Senkou_Span_A'] = senkou_span_a
    df['Senkou_Span_B'] = senkou_span_b
    return df
