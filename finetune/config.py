import os

class Config:
    """
    Configuration class for the entire project.
    """

    def __init__(self):
        # =================================================================
        # Data & Feature Parameters
        # =================================================================
        # TODO: Update this path to your Qlib data directory.
        self.qlib_data_path = "../qlib_data/dataForPKU-2/"
        self.instrument = 'all'
        
        self.run_id = "zmj_test_20260429_new-12-features-14bit_new-predictor-seqlen120_predict_30"

        # Overall time range for data loading from Qlib.
        self.dataset_begin_time = "2005-01-04"
        self.dataset_end_time = '2025-11-07'

        # Sliding window parameters for creating samples.
        self.lookback_window = 120  # Number of past time steps for input.
        self.predict_window = 30  # Number of future time steps for prediction.
        self.max_context = 512  # Maximum context length for the model.

        # Features to be used from the raw data.
        self.feature_list = [
            "open", "high", "low", "close",
            "vol", "amt",
            "flag",
            "circulating_market_cap",
            "open_return", "close_return",
            "hy_open_return", "hy_close_return",
        ]
        # Time-based features to be generated.
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']

        # =================================================================
        # Dataset Splitting & Paths
        # =================================================================
        # Note: The validation/test set starts earlier than the training/validation set ends
        # to account for the `lookback_window`.
        self.train_time_range = ["2005-01-04", "2022-12-31"]
        self.val_time_range = ["2022-09-01", "2024-06-30"]
        self.test_time_range = ["2024-04-01", "2025-11-07"]
        self.backtest_time_range = ["2024-07-01", "2025-11-06"]

        # TODO: Directory to save the processed, pickled datasets.
        self.dataset_path = "./data/processed_datasets_110"
        # self.dataset_path = "./data/processed_datasets_cleaned_6"

        # =================================================================
        # Training Hyperparameters
        # =================================================================
        self.clip = 5.0  # Clipping value for normalized data to prevent outliers.

        self.epochs = 250
        self.log_interval = 100  # Log training status every N batches.
        self.batch_size = 250  # Batch size per GPU.

        # Early stopping patience: stop if val loss does not improve for N epochs.
        self.early_stop_patience = 999999

        # Number of samples to draw for one "epoch" of training/validation.
        # This is useful for large datasets where a true epoch is too long.
        self.n_train_iter = 20000 * self.batch_size
        self.n_val_iter = 400 * self.batch_size

        # Learning rates for different model components.
        self.tokenizer_learning_rate = 1e-5
        self.predictor_learning_rate = 4e-5

        # Gradient accumulation to simulate a larger batch size.
        self.accumulation_steps = 1

        # AdamW optimizer parameters.
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_weight_decay = 0.1

        # Miscellaneous
        self.seed = 100  # Global random seed for reproducibility.
        
        self.tokenizer_model_initialize_params = {
            "attn_dropout_p": 0.0,
            "beta": 0.05,
            "d_in": 12,
            "d_model": 512,
            "ff_dim": 1024,
            "ffn_dropout_p": 0.0,
            "gamma": 1.1,
            "gamma0": 1.0,
            "group_size": 4,
            "n_dec_layers": 5,
            "n_enc_layers": 5,
            "n_heads": 8,
            "resid_dropout_p": 0.0,
            "s1_bits": 14,
            "s2_bits": 14,
            "zeta": 0.05
        } # base predictor 的参数
        self.predictor_model_initialize_params = {
            "attn_dropout_p": 0.1,
            "d_model": 512,
            "ff_dim": 1024,
            "ffn_dropout_p": 0.25,
            "learn_te": True,
            "n_heads": 8,
            "n_layers": 8,
            "resid_dropout_p": 0.25,
            "s1_bits": 14,
            "s2_bits": 14,
            "token_dropout_p": 0.1
        }

        # =================================================================
        # Experiment Logging & Saving
        # =================================================================
        self.use_comet = True # Set to False if you don't want to use Comet ML
        self.comet_config = {
            # It is highly recommended to load secrets from environment variables
            # for security purposes. Example: os.getenv("COMET_API_KEY")
            "api_key": "",
            "project_name": f"Kronos-Finetune-Demo-{self.run_id}",
            "workspace": "kronos-kcl" # TODO: Change to your Comet ML workspace name
        }
        self.comet_tag = 'finetune_demo'
        self.comet_name = 'finetune_demo'

        # Base directory for saving model checkpoints and results.
        # Using a general 'outputs' directory is a common practice.
        self.save_path = "./outputs/models"
        self.tokenizer_save_folder_name = f'finetune_tokenizer_demo_{self.run_id}'
        self.predictor_save_folder_name = f'finetune_predictor_demo_{self.run_id}'
        self.backtest_save_folder_name = f'finetune_backtest_demo_{self.run_id}'

        # Path for backtesting results.
        self.backtest_result_path = "./outputs/backtest_results"

        # =================================================================
        # Model & Checkpoint Paths
        # =================================================================
        # TODO: Update these paths to your pretrained model locations.
        # These can be local paths or Hugging Face Hub model identifiers.
        self.pretrained_tokenizer_path = "NeoQuasar/Kronos-Tokenizer-base"
        self.pretrained_predictor_path = "NeoQuasar/Kronos-small"

        # Paths to the fine-tuned models, derived from the save_path.
        # These will be generated automatically during training.
        self.finetuned_tokenizer_path = f"{self.save_path}/{self.tokenizer_save_folder_name}/checkpoints/best_model"
        self.finetuned_predictor_path = f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"

        self.n_fast_test_iter = self.batch_size * 1000
        
        # =================================================================
        # Backtesting Parameters
        # =================================================================
        self.backtest_n_symbol_hold = 50  # Number of symbols to hold in the portfolio.
        self.backtest_n_symbol_drop = 5  # Number of symbols to drop from the pool.
        self.backtest_hold_thresh = 5  # Minimum holding period for a stock.
        self.inference_T = 0.6
        self.inference_top_p = 0.9
        self.inference_top_k = 0
        self.inference_sample_count = 5
        self.backtest_batch_size = 1000
        self.backtest_benchmark = self._set_benchmark(self.instrument)
        
        # =================================================================
        # New added by ZMJ
        # =================================================================
        self.backtest_pred = 'close_return'    # supported: 'close' or 'close_return' now, 20260112
        if self.backtest_pred == 'close_return':
            self.backtest_save_folder_name += '_closereturn'

    def _set_benchmark(self, instrument):
        dt_benchmark = {
            'csi800': "SH000906",
            'csi1000': "SH000852",
            'csi300': "SH000300",
            'all': '000906',
        }
        if instrument in dt_benchmark:
            return dt_benchmark[instrument]
        else:
            raise ValueError(f"Benchmark not defined for instrument: {instrument}")
