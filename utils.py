import logging
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

def setup_logging(name: str) -> logging.Logger:
    """Set up logging configuration"""
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    return logging.getLogger(name)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: Path
) -> str:
    """Save model checkpoint"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch{epoch}_{timestamp}.pt"
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    
    return str(checkpoint_path)

def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None
) -> Dict[str, Any]:
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint

def format_html(html_string: str) -> str:
    """Format HTML string for better readability"""
    import html
    from bs4 import BeautifulSoup
    
    # Decode HTML entities
    decoded_html = html.unescape(html_string)
    
    # Pretty format
    soup = BeautifulSoup(decoded_html, 'html.parser')
    return soup.prettify()