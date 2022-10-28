import pandas as pd
import numpy as np

def sharp(ret_sr, freq):
    mean = ret_sr.mean()
    std = ret_sr.std()
    return mean / std * np.sqrt(250 / freq)


def anual_ret(ret_sr, freq):
    netValue = (1 + ret_sr).prod()
    n = len(ret_sr) * freq / 250
    r = netValue ** (1 / n) - 1
    return r


def max_draw(ret_sr):
    netValue = (1 + ret_sr).cumprod()
    maxValue = netValue.cummax()
    drawDown = (maxValue - netValue) / maxValue
    return max(drawDown)


def win(pred, true):
    return sum(np.array(pred) == np.array(true)) / len(pred)


def netValue(ret_sr, index_sr, save=True):
    d = {
        'strategy': (1 + ret_sr).cumprod(),
        'hold': (1 + index_sr).cumprod()
    }

    netValue = pd.DataFrame(d)
    ax = netValue.plot(title='Net value', figsize=(20, 5))
    fig = ax.get_figure()
    if save:
        fig.savefig('netValue.jpg')

