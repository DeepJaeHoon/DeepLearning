import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates

def show_candle_chart(input, output, ground_truth):
    input_open = input[:, :, 0].flatten()
    input_close = input[:, :, 4].flatten()
    input_low = input[:, :, 2].flatten()
    input_high = input[:, :, 1].flatten()
    input_volume = input[:, :, 3].flatten()

    pred_open = output[:, :, 0].flatten()
    pred_close = output[:, :, 4].flatten()
    pred_low = output[:, :, 2].flatten()
    pred_high = output[:, :, 1].flatten()
    pred_volume = output[:, :, 3].flatten()

    gt_open = ground_truth[:, :, 0].flatten()
    gt_close = ground_truth[:, :, 4].flatten()
    gt_low = ground_truth[:, :, 2].flatten()
    gt_high = ground_truth[:, :, 1].flatten()
    gt_volume = ground_truth[:, :, 3].flatten()

    def create_candlestick_data(dates, open_prices, high_prices, low_prices, close_prices):
        candlestick_data = []
        for i in range(len(open_prices)):
            candlestick_data.append([dates[i], open_prices[i], high_prices[i], low_prices[i], close_prices[i]])
        return candlestick_data

    dates = np.arange(len(input_open))

    input_data = create_candlestick_data(dates, input_open, input_high, input_low, input_close)
    pred_data = create_candlestick_data(dates, pred_open, pred_high, pred_low, pred_close)
    gt_data = create_candlestick_data(dates, gt_open, gt_high, gt_low, gt_close)

    fig, ax = plt.subplots(figsize=(14, 7))

    candlestick_ohlc(ax, input_data, colorup='g', colordown='r', width=0.6)

    def animate(i):
        ax.clear()
        input_start = max(0, i - 6)
        input_end = i + 1

        pred_start = i + 1
        pred_end = min(len(dates), i + 4)

        if input_start < input_end:
            candlestick_ohlc(ax, input_data[input_start:input_end], colorup='b', colordown='r', width=0.6)

        if i >= 6:
            if pred_start < pred_end:
                candlestick_ohlc(ax, pred_data[pred_start:pred_end], colorup='purple', colordown='orange', width=0.6)
                candlestick_ohlc(ax, gt_data[pred_start:pred_end], colorup='purple', colordown='orange', width=0.6, alpha=0.3)

        ax.set_title('Candlestick Chart with Input, Predicted, and Ground Truth Data')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.set_xlim(input_start, min(len(dates) - 1, pred_end - 1))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    ani = animation.FuncAnimation(fig, animate, frames=len(input_data), repeat=False)
    plt.show()
