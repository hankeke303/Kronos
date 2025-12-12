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
        self.qlib_data_path = "../qlib_data/dataForPKU/"
        self.instrument = 'all'
        
        # self.run_id = "test_20251118_1806_scratch_new-dataset"
        # self.run_id = "test_20251121_1209_epochs-250_batch-400_lr1e-4"
        # self.run_id = "test_20251122_0933_new-model-params"
        # self.run_id = "test_20251122_1928_epochs-250_batch-400_lr1e-4_cosinelr"
        # self.run_id = "test_20251122_2019_new-model-params-2"
        # self.run_id = "test_20251125_2327_new-model-params-5-2"
        # self.run_id = "test_20251126_0034_train_predictor_with_new_tokenizer"
        # self.run_id = "test_20251126_0918_train_predictor_with_new_tokenizer_scratch"
        # self.run_id = "test_202512011451_predictor_w_tokenizer11301359_l0072-2"
        self.run_id = "test_202512041503_predictor_12feature_14bit-2_tokloss0.0059"

        # Overall time range for data loading from Qlib.
        self.dataset_begin_time = "2005-01-04"
        self.dataset_end_time = '2025-11-07'

        # Sliding window parameters for creating samples.
        self.lookback_window = 90  # Number of past time steps for input.
        self.predict_window = 10  # Number of future time steps for prediction.
        self.max_context = 512  # Maximum context length for the model.

        # Features to be used from the raw data.
        # self.feature_list = ['open', 'high', 'low', 'close', 'vol', 'amt']
        self.feature_list = [
            "open", "high", "low", "close",
            # "high_limit", "low_limit",
            "vol", "amt",
            "ma_tt_5", "ma_tt_10", # "ma_tt_20", "ma_tt_60", "ma_tt_120",
            "rsi_tt_3", "rsi_tt_6", # "rsi_tt_12", "rsi_tt_14",
            # "macd_tt_dif", "macd_tt_dea", "macd_tt_macd",
            "macd_tt_dif", "macd_tt_macd",
            # "flag", "is_st"
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
        self.dataset_path = "./data/processed_datasets_origin"
        # self.dataset_path = "./data/processed_datasets_cleaned_6"

        # =================================================================
        # Training Hyperparameters
        # =================================================================
        self.clip = 5.0  # Clipping value for normalized data to prevent outliers.

        self.epochs = 250
        self.log_interval = 100  # Log training status every N batches.
        self.batch_size = 400  # Batch size per GPU.

        # Number of samples to draw for one "epoch" of training/validation.
        # This is useful for large datasets where a true epoch is too long.
        self.n_train_iter = 20000 * self.batch_size
        self.n_val_iter = 400 * self.batch_size

        # Learning rates for different model components.
        self.tokenizer_learning_rate = 5e-6
        self.predictor_learning_rate = 4e-5

        # Gradient accumulation to simulate a larger batch size.
        self.accumulation_steps = 1

        # AdamW optimizer parameters.
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_weight_decay = 0.4

        # Miscellaneous
        self.seed = 100  # Global random seed for reproducibility.
        
        # self.tokenizer_model_initialize_params = {
        #     "attn_dropout_p": 0.0,
        #     "beta": 0.05,
        #     "d_in": 20,
        #     "d_model": 256,
        #     "ff_dim": 512,
        #     "ffn_dropout_p": 0.0,
        #     "gamma": 1.1,
        #     "gamma0": 1.0,
        #     "group_size": 4,
        #     "n_dec_layers": 4,
        #     "n_enc_layers": 4,
        #     "n_heads": 4,
        #     "resid_dropout_p": 0.0,
        #     "s1_bits": 10,
        #     "s2_bits": 10,
        #     "zeta": 0.05
        # }
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
        }
        # self.predictor_model_initialize_params = {
        #     "attn_dropout_p": 0.0,
        #     "d_model": 832,
        #     "ff_dim": 2048,
        #     "ffn_dropout_p": 0.2,
        #     "learn_te": True,
        #     "n_heads": 16,
        #     "n_layers": 12,
        #     "resid_dropout_p": 0.2,
        #     "s1_bits": 10,
        #     "s2_bits": 10,
        #     "token_dropout_p": 0.0
        # } # base predictor 的参数
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
            "api_key": "yjtiDQADcsqBhdN5N6iXeZNpd",
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
        # self.pretrained_tokenizer_path = "./outputs/models/finetune_tokenizer_demo_test_20251202_1352_tokenizer_12feature_14bit/checkpoints/best_model"
        self.pretrained_predictor_path = "NeoQuasar/Kronos-small"

        # Paths to the fine-tuned models, derived from the save_path.
        # These will be generated automatically during training.
        # self.finetuned_tokenizer_path = f"{self.save_path}/{self.tokenizer_save_folder_name}/checkpoints/best_model"
        # self.finetuned_tokenizer_path = "NeoQuasar/Kronos-Tokenizer-base"
        self.finetuned_tokenizer_path = "./outputs/models/finetune_tokenizer_demo_test_20251202_1352_tokenizer_12feature_14bit/checkpoints/best_model"
        # self.finetuned_predictor_path = f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"
        # self.finetuned_predictor_path = "NeoQuasar/Kronos-base"
        self.finetuned_predictor_path = "./outputs/models/finetune_predictor_demo_test_20251126_0918_train_predictor_with_new_tokenizer_scratch/checkpoints/best_model"

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
