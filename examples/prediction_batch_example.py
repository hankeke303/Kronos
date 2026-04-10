import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from model import Kronos, KronosTokenizer, KronosPredictor


def plot_prediction(kline_df, pred_df):
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Ground Truth'
    sr_pred_close.name = "Prediction"

    sr_volume = kline_df['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = 'Ground Truth'
    sr_pred_volume.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(close_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax1.plot(close_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax1.set_ylabel('Close Price', fontsize=14)
    ax1.legend(loc='lower left', fontsize=12)
    ax1.grid(True)

    ax2.plot(volume_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax2.plot(volume_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax2.set_ylabel('Volume', fontsize=14)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Kronos batch prediction example")
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=os.getenv("KRONOS_TOKENIZER_PATH", "NeoQuasar/Kronos-Tokenizer-base"),
        help="Tokenizer path or Hugging Face repo id",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.getenv("KRONOS_MODEL_PATH", "NeoQuasar/Kronos-base"),
        help="Model path or Hugging Face repo id",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=os.path.join(CURRENT_DIR, "data", "XSHG_5min_600977.csv"),
        help="Input csv path",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device")
    parser.add_argument("--lookback", type=int, default=400, help="Lookback length")
    parser.add_argument("--pred-len", type=int, default=120, help="Prediction length")
    parser.add_argument("--batch-count", type=int, default=5, help="How many rolling windows to run")
    args = parser.parse_args()

    # 1. Load model and tokenizer
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer_path)
    model = Kronos.from_pretrained(args.model_path)

    # 2. Instantiate predictor
    predictor = KronosPredictor(model, tokenizer, device=args.device, max_context=512)

    # 3. Prepare data
    df = pd.read_csv(args.data_path)

    if 'timestamps' not in df.columns and 'datetime' in df.columns:
        df['timestamps'] = df['datetime']
    df['timestamps'] = pd.to_datetime(df['timestamps'])

    # 兼容不同数据命名：volume/amount 或 vol/amt
    if 'volume' not in df.columns and 'vol' in df.columns:
        df['volume'] = df['vol']
    if 'amount' not in df.columns and 'amt' in df.columns:
        df['amount'] = df['amt']

    lookback = args.lookback
    pred_len = args.pred_len

    dfs = []
    xtsp = []
    ytsp = []
    for i in range(args.batch_count):
        start = i * lookback
        end = start + lookback - 1
        pred_start = end + 1
        pred_end = pred_start + pred_len - 1

        idf = df.loc[start:end, ['open', 'high', 'low', 'close', 'volume', 'amount']]
        i_x_timestamp = df.loc[start:end, 'timestamps']
        i_y_timestamp = df.loc[pred_start:pred_end, 'timestamps']

        if len(idf) < lookback or len(i_y_timestamp) < pred_len:
            break

        dfs.append(idf)
        xtsp.append(i_x_timestamp)
        ytsp.append(i_y_timestamp)

    if not dfs:
        raise ValueError("No valid windows found. Check data length, lookback and pred_len.")

    pred_df = predictor.predict_batch(
        df_list=dfs,
        x_timestamp_list=xtsp,
        y_timestamp_list=ytsp,
        pred_len=pred_len,
    )

    print(pred_df)


if __name__ == '__main__':
    main()
