"""
Checkpoint management system for Khmer OCR Seq2Seq model.
Handles model saving, loading, and training resumption.
"""
import os
import torch
import shutil
import glob
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging


class CheckpointManager:
    """
    Checkpoint manager for model training state persistence.
    
    Features:
    - Automatic checkpoint saving with epoch numbering
    - Best model tracking based on validation metrics
    - Training resumption from saved checkpoints
    - Cleanup of old checkpoints to save disk space
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "models/checkpoints",
        max_checkpoints: int = 5,
        save_best: bool = True
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_best: Whether to save best model separately
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Best model path
        self.best_model_path = self.checkpoint_dir / "best_model.pth"
        
        # Setup logging
        self.logger = logging.getLogger("CheckpointManager")
        self.logger.setLevel(logging.INFO)
    
    def save_checkpoint(
        self,
        checkpoint_data: Dict[str, Any],
        epoch: int,
        is_best: bool = False,
        filename: Optional[str] = None
    ) -> str:
        """
        Save model checkpoint to disk.
        
        Args:
            checkpoint_data: Dictionary containing model state and metadata
            epoch: Current epoch number
            is_best: Whether this is the best model so far
            filename: Custom filename (optional)
            
        Returns:
            Path to saved checkpoint file
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:03d}.pth"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Add timestamp to checkpoint data
        import datetime
        checkpoint_data['timestamp'] = datetime.datetime.now().isoformat()
        checkpoint_data['checkpoint_path'] = str(checkpoint_path)
        
        try:
            # Save checkpoint
            torch.save(checkpoint_data, checkpoint_path)
            self.logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Save best model if needed
            if is_best and self.save_best:
                shutil.copy2(checkpoint_path, self.best_model_path)
                self.logger.info(f"Saved best model: {self.best_model_path}")
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
            return str(checkpoint_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint {checkpoint_path}: {e}")
            raise e
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load checkpoint from disk.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary containing model state and metadata
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            # Load checkpoint (weights_only=False for compatibility with config objects)
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            self.logger.info(f"Loaded checkpoint: {checkpoint_path}")
            
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            raise e
    
    def load_best_model(self) -> Optional[Dict[str, Any]]:
        """
        Load the best saved model.
        
        Returns:
            Dictionary containing best model state, or None if not found
        """
        if self.best_model_path.exists():
            return self.load_checkpoint(str(self.best_model_path))
        else:
            self.logger.warning("Best model not found")
            return None
    
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load the most recent checkpoint.
        
        Returns:
            Dictionary containing latest checkpoint state, or None if not found
        """
        checkpoint_files = self.list_checkpoints()
        
        if not checkpoint_files:
            self.logger.info("No checkpoints found")
            return None
        
        # Get most recent checkpoint (highest epoch number)
        latest_checkpoint = max(checkpoint_files, key=lambda x: x['epoch'])
        return self.load_checkpoint(latest_checkpoint['path'])
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.
        
        Returns:
            List of dictionaries with checkpoint information
        """
        checkpoint_pattern = self.checkpoint_dir / "checkpoint_epoch_*.pth"
        checkpoint_files = glob.glob(str(checkpoint_pattern))
        
        checkpoints = []
        for checkpoint_file in checkpoint_files:
            try:
                # Extract epoch number from filename
                filename = Path(checkpoint_file).name
                epoch_str = filename.replace("checkpoint_epoch_", "").replace(".pth", "")
                epoch = int(epoch_str)
                
                # Get file size and modification time
                stat = os.stat(checkpoint_file)
                size_mb = stat.st_size / (1024 * 1024)
                mtime = stat.st_mtime
                
                checkpoints.append({
                    'path': checkpoint_file,
                    'epoch': epoch,
                    'size_mb': size_mb,
                    'mtime': mtime,
                    'filename': filename
                })
                
            except (ValueError, OSError) as e:
                self.logger.warning(f"Skipping invalid checkpoint file {checkpoint_file}: {e}")
        
        # Sort by epoch number
        checkpoints.sort(key=lambda x: x['epoch'])
        return checkpoints
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints to free disk space."""
        if self.max_checkpoints <= 0:
            return
        
        checkpoints = self.list_checkpoints()
        
        # Keep only the most recent checkpoints
        if len(checkpoints) > self.max_checkpoints:
            # Sort by epoch (most recent first)
            checkpoints.sort(key=lambda x: x['epoch'], reverse=True)
            
            # Remove old checkpoints
            for checkpoint in checkpoints[self.max_checkpoints:]:
                try:
                    os.remove(checkpoint['path'])
                    self.logger.info(f"Removed old checkpoint: {checkpoint['filename']}")
                except OSError as e:
                    self.logger.warning(f"Failed to remove checkpoint {checkpoint['path']}: {e}")
    
    def delete_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Delete a specific checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint to delete
            
        Returns:
            True if successfully deleted, False otherwise
        """
        try:
            os.remove(checkpoint_path)
            self.logger.info(f"Deleted checkpoint: {checkpoint_path}")
            return True
        except OSError as e:
            self.logger.error(f"Failed to delete checkpoint {checkpoint_path}: {e}")
            return False
    
    def get_checkpoint_info(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Get information about a checkpoint without loading the full model.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary with checkpoint metadata
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            # Load only the metadata (not the full model)
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            info = {
                'epoch': checkpoint.get('epoch', 'unknown'),
                'metrics': checkpoint.get('metrics', {}),
                'best_cer': checkpoint.get('best_cer', 'unknown'),
                'global_step': checkpoint.get('global_step', 'unknown'),
                'timestamp': checkpoint.get('timestamp', 'unknown'),
                'config': checkpoint.get('config', {})
            }
            
            # Add file information
            stat = os.stat(checkpoint_path)
            info['file_size_mb'] = stat.st_size / (1024 * 1024)
            info['file_path'] = str(checkpoint_path)
            
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to get checkpoint info {checkpoint_path}: {e}")
            raise e
    
    def backup_best_model(self, backup_dir: str = "models/backups") -> Optional[str]:
        """
        Create a backup copy of the best model.
        
        Args:
            backup_dir: Directory to save backup
            
        Returns:
            Path to backup file, or None if best model doesn't exist
        """
        if not self.best_model_path.exists():
            self.logger.warning("Best model not found for backup")
            return None
        
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Create backup filename with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"best_model_backup_{timestamp}.pth"
        backup_path = backup_dir / backup_filename
        
        try:
            shutil.copy2(self.best_model_path, backup_path)
            self.logger.info(f"Created backup: {backup_path}")
            return str(backup_path)
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return None
    
    def get_disk_usage(self) -> Dict[str, float]:
        """
        Get disk usage information for checkpoints.
        
        Returns:
            Dictionary with disk usage statistics
        """
        checkpoints = self.list_checkpoints()
        
        total_size = sum(cp['size_mb'] for cp in checkpoints)
        avg_size = total_size / len(checkpoints) if checkpoints else 0
        
        # Include best model if it exists
        best_model_size = 0
        if self.best_model_path.exists():
            best_model_size = os.path.getsize(self.best_model_path) / (1024 * 1024)
        
        return {
            'total_checkpoints': len(checkpoints),
            'total_size_mb': total_size,
            'average_size_mb': avg_size,
            'best_model_size_mb': best_model_size,
            'total_disk_usage_mb': total_size + best_model_size
        } 