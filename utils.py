import logging
from datetime import datetime
from pathlib import Path

def setup_logger(output_dir='results', name='training'):
    """Setup file + console logging"""
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = Path(output_dir) / f'{name}_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(name)