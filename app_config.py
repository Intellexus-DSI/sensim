import yaml
import os

CONFIG_FILE_NAME = 'config.yaml'

class AppConfig:
    _instance = None

    # Define your defaults directly in the class
    DEFAULTS = {
        "config_path": None,
        "scoring_script_dir": "/home/shay/Best-Worst-Scaling-Scripts",
        "scoring_script_file_name": "get-scores-from-BWS-annotations-counting.pl",
    }

    def __new__(cls, *args, **kwargs):
        """
        Singleton Pattern: Ensures config is loaded from disk only once.
        """
        if cls._instance is None:
            cls._instance = super(AppConfig, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, yaml_path=CONFIG_FILE_NAME):
        """
        Initializes the configuration.
        """
        if self._initialized:
            return

        self.yaml_path = yaml_path
        self._config = self.DEFAULTS.copy()
        self._load_from_file()
        self._apply_env_overrides()

        self._initialized = True

    def _load_from_file(self):
        """
        Internal method to handle the file loading and merging.
        """
        if os.path.exists(self.yaml_path):
            # print(f"Loading configuration from {self.yaml_path}...")
            try:
                with open(self.yaml_path, 'r') as f:
                    file_settings = yaml.safe_load(f)
                    if file_settings:
                        self._config.update(file_settings)
            except Exception as e:
                print(f"Error loading config file: {e}. Using defaults.")
        else:
            print(f"Warning: Config file '{self.yaml_path}' not found. Using defaults.")

    def _apply_env_overrides(self):
        """Apply environment variable overrides from config."""
        hf_cache_dir = self._config.get('hf_cache_dir')
        if hf_cache_dir:
            os.environ['HF_HOME'] = hf_cache_dir

        cuda_visible = self._config.get('cuda_visible_devices')
        if cuda_visible is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_visible)

    def get(self, key, default=None):
        """Safe getter for config values."""
        return self._config.get(key, default)

    def __getitem__(self, key):
        """Allows dictionary-style access: config['key']"""
        return self._config[key]

    @property
    def all(self):
        """Returns the full configuration dictionary."""
        return self._config

    def reload(self, yaml_path=None):
        """Forces a reload of the config file, optionally from a new path."""
        if yaml_path is not None:
            self.yaml_path = yaml_path
        print(f"Reloading configuration from {self.yaml_path}...")
        self._config = self.DEFAULTS.copy()
        self._load_from_file()
        self._apply_env_overrides()

    def reload_cuda_visible_devices(self):
        """Re-read only cuda_visible_devices from the config file and apply it to the environment."""
        try:
            with open(self.yaml_path, 'r') as f:
                file_settings = yaml.safe_load(f) or {}
            cuda_visible = file_settings.get('cuda_visible_devices')
            if cuda_visible is not None:
                cuda_visible = str(cuda_visible)
                os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible
                self._config['cuda_visible_devices'] = cuda_visible
        except Exception as e:
            print(f"Warning: could not reload cuda_visible_devices: {e}")