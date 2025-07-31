import logging
import os
import sys
import time
import json
import pickle
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from contextlib import contextmanager
from threading import Lock

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CONFIG_FILE = 'config.json'
DEFAULT_CONFIG = {
    'seed': 42,
    'log_level': 'INFO',
    'log_file': 'utils.log',
    'data_dir': 'data',
    'model_dir': 'models'
}

# Enum for log levels
class LogLevel(Enum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40

# Dataclass for configuration
@dataclass
class Config:
    seed: int
    log_level: str
    log_file: str
    data_dir: str
    model_dir: str

# Context manager for logging
class LogContext:
    def __init__(self, level: LogLevel):
        self.level = level

    def __enter__(self):
        logger.setLevel(self.level.value)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.setLevel(logging.INFO)

# Utility functions
class Utils(ABC):
    def __init__(self, config: Config):
        self.config = config
        self.lock = Lock()

    @abstractmethod
    def load_config(self) -> Config:
        pass

    @abstractmethod
    def save_config(self, config: Config) -> None:
        pass

    def load_json(self, file_path: str) -> Dict:
        with open(file_path, 'r') as f:
            return json.load(f)

    def save_json(self, data: Dict, file_path: str) -> None:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def load_pickle(self, file_path: str) -> Any:
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def save_pickle(self, data: Any, file_path: str) -> None:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    def get_timestamp(self) -> str:
        return time.strftime('%Y-%m-%d %H:%M:%S')

    def get_logger(self) -> logging.Logger:
        return logger

    def log(self, level: LogLevel, message: str) -> None:
        self.get_logger().log(level.value, message)

    def log_debug(self, message: str) -> None:
        self.log(LogLevel.DEBUG, message)

    def log_info(self, message: str) -> None:
        self.log(LogLevel.INFO, message)

    def log_warning(self, message: str) -> None:
        self.log(LogLevel.WARNING, message)

    def log_error(self, message: str) -> None:
        self.log(LogLevel.ERROR, message)

    def get_config(self) -> Config:
        return self.config

    def set_config(self, config: Config) -> None:
        self.config = config

    def get_lock(self) -> Lock:
        return self.lock

    def acquire_lock(self) -> None:
        self.get_lock().acquire()

    def release_lock(self) -> None:
        self.get_lock().release()

# Helper functions
def get_config() -> Config:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
    else:
        config = DEFAULT_CONFIG
    return Config(**config)

def save_config(config: Config) -> None:
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config.__dict__, f, indent=4)

def load_json(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data: Dict, file_path: str) -> None:
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def load_pickle(file_path: str) -> Any:
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data: Any, file_path: str) -> None:
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def get_timestamp() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')

def get_logger() -> logging.Logger:
    return logger

def log(level: LogLevel, message: str) -> None:
    get_logger().log(level.value, message)

def log_debug(message: str) -> None:
    log(LogLevel.DEBUG, message)

def log_info(message: str) -> None:
    log(LogLevel.INFO, message)

def log_warning(message: str) -> None:
    log(LogLevel.WARNING, message)

def log_error(message: str) -> None:
    log(LogLevel.ERROR, message)

def get_config() -> Config:
    return get_config()

def set_config(config: Config) -> None:
    save_config(config)

def get_lock() -> Lock:
    return Utils().get_lock()

def acquire_lock() -> None:
    get_lock().acquire()

def release_lock() -> None:
    get_lock().release()

# Unit tests
import unittest
from unittest.mock import Mock

class TestUtils(unittest.TestCase):
    def test_get_config(self):
        config = get_config()
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.log_level, 'INFO')
        self.assertEqual(config.log_file, 'utils.log')
        self.assertEqual(config.data_dir, 'data')
        self.assertEqual(config.model_dir, 'models')

    def test_save_config(self):
        config = Config(seed=42, log_level='DEBUG', log_file='utils_debug.log', data_dir='data_debug', model_dir='models_debug')
        save_config(config)
        with open(CONFIG_FILE, 'r') as f:
            loaded_config = json.load(f)
        self.assertEqual(loaded_config['seed'], 42)
        self.assertEqual(loaded_config['log_level'], 'DEBUG')
        self.assertEqual(loaded_config['log_file'], 'utils_debug.log')
        self.assertEqual(loaded_config['data_dir'], 'data_debug')
        self.assertEqual(loaded_config['model_dir'], 'models_debug')

    def test_load_json(self):
        data = {'key': 'value'}
        file_path = 'test.json'
        save_json(data, file_path)
        loaded_data = load_json(file_path)
        self.assertEqual(loaded_data['key'], 'value')
        os.remove(file_path)

    def test_save_json(self):
        data = {'key': 'value'}
        file_path = 'test.json'
        save_json(data, file_path)
        with open(file_path, 'r') as f:
            loaded_data = json.load(f)
        self.assertEqual(loaded_data['key'], 'value')
        os.remove(file_path)

    def test_load_pickle(self):
        data = {'key': 'value'}
        file_path = 'test.pkl'
        save_pickle(data, file_path)
        loaded_data = load_pickle(file_path)
        self.assertEqual(loaded_data['key'], 'value')
        os.remove(file_path)

    def test_save_pickle(self):
        data = {'key': 'value'}
        file_path = 'test.pkl'
        save_pickle(data, file_path)
        with open(file_path, 'rb') as f:
            loaded_data = pickle.load(f)
        self.assertEqual(loaded_data['key'], 'value')
        os.remove(file_path)

    def test_get_timestamp(self):
        timestamp = get_timestamp()
        self.assertRegex(timestamp, r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$')

    def test_get_logger(self):
        logger = get_logger()
        self.assertIsInstance(logger, logging.Logger)

    def test_log(self):
        log(LogLevel.INFO, 'Test log message')
        self.assertIn('INFO', get_logger().log)

    def test_log_debug(self):
        log_debug('Test debug message')
        self.assertIn('DEBUG', get_logger().log)

    def test_log_info(self):
        log_info('Test info message')
        self.assertIn('INFO', get_logger().log)

    def test_log_warning(self):
        log_warning('Test warning message')
        self.assertIn('WARNING', get_logger().log)

    def test_log_error(self):
        log_error('Test error message')
        self.assertIn('ERROR', get_logger().log)

    def test_get_config(self):
        config = get_config()
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.log_level, 'INFO')
        self.assertEqual(config.log_file, 'utils.log')
        self.assertEqual(config.data_dir, 'data')
        self.assertEqual(config.model_dir, 'models')

    def test_set_config(self):
        config = Config(seed=42, log_level='DEBUG', log_file='utils_debug.log', data_dir='data_debug', model_dir='models_debug')
        set_config(config)
        with open(CONFIG_FILE, 'r') as f:
            loaded_config = json.load(f)
        self.assertEqual(loaded_config['seed'], 42)
        self.assertEqual(loaded_config['log_level'], 'DEBUG')
        self.assertEqual(loaded_config['log_file'], 'utils_debug.log')
        self.assertEqual(loaded_config['data_dir'], 'data_debug')
        self.assertEqual(loaded_config['model_dir'], 'models_debug')

    def test_get_lock(self):
        lock = get_lock()
        self.assertIsInstance(lock, Lock)

    def test_acquire_lock(self):
        lock = get_lock()
        acquire_lock()
        self.assertTrue(lock.locked())

    def test_release_lock(self):
        lock = get_lock()
        acquire_lock()
        release_lock()
        self.assertFalse(lock.locked())

if __name__ == '__main__':
    unittest.main()