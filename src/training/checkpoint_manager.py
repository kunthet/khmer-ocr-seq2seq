"""
Checkpoint management system for Khmer OCR Seq2Seq model.
Handles model saving, loading, and training resumption with Google Drive backup.
"""
import os
import torch
import shutil
import glob
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
import json


class CheckpointManager:
    """
    Checkpoint manager for model training state persistence with Google Drive backup.
    
    Features:
    - Automatic checkpoint saving with epoch numbering
    - Best model tracking based on validation metrics
    - Training resumption from saved checkpoints
    - Google Drive backup for checkpoints
    - Cleanup of old checkpoints to save disk space
    - Smart fallback from Google Drive if local checkpoints missing
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "models/checkpoints",
        max_checkpoints: int = 5,
        save_best: bool = True,
        gdrive_backup: bool = True,
        gdrive_dir: str = "/content/drive/MyDrive/KhmerOCR_Checkpoints"
    ):
        """
        Initialize checkpoint manager with optional Google Drive backup.
        
        Args:
            checkpoint_dir: Local directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep locally
            save_best: Whether to save best model separately
            gdrive_backup: Whether to backup to Google Drive
            gdrive_dir: Google Drive directory for backups
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.gdrive_backup = gdrive_backup
        self.gdrive_dir = Path(gdrive_dir) if gdrive_backup else None
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Google Drive directory if backup enabled
        if self.gdrive_backup and self.gdrive_dir:
            try:
                self.gdrive_dir.mkdir(parents=True, exist_ok=True)
                (self.gdrive_dir / "models").mkdir(exist_ok=True)
                (self.gdrive_dir / "logs").mkdir(exist_ok=True)
                self.logger = logging.getLogger("CheckpointManager")
                self.logger.info(f"Google Drive backup enabled: {self.gdrive_dir}")
            except Exception as e:
                self.logger = logging.getLogger("CheckpointManager")
                self.logger.warning(f"Google Drive backup disabled due to error: {e}")
                self.gdrive_backup = False
        
        # Best model paths
        self.best_model_path = self.checkpoint_dir / "best_model.pth"
        self.gdrive_best_model_path = (self.gdrive_dir / "models" / "best_model.pth") if self.gdrive_backup else None
        
        # Setup logging
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger("CheckpointManager")
            self.logger.setLevel(logging.INFO)
    
    def save_checkpoint(
        self,
        checkpoint_data: Dict[str, Any],
        epoch: int,
        is_best: bool = False,
        filename: Optional[str] = None,
        backup_to_gdrive: bool = True
    ) -> str:
        """
        Save model checkpoint to disk and optionally to Google Drive.
        
        Args:
            checkpoint_data: Dictionary containing model state and metadata
            epoch: Current epoch number
            is_best: Whether this is the best model so far
            filename: Custom filename (optional)
            backup_to_gdrive: Whether to backup this checkpoint to Google Drive
            
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
            # Save checkpoint locally
            torch.save(checkpoint_data, checkpoint_path)
            self.logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Save best model if needed
            if is_best and self.save_best:
                shutil.copy2(checkpoint_path, self.best_model_path)
                self.logger.info(f"Saved best model: {self.best_model_path}")
                
                # Backup best model to Google Drive
                if self.gdrive_backup and self.gdrive_best_model_path:
                    try:
                        shutil.copy2(checkpoint_path, self.gdrive_best_model_path)
                        self.logger.info(f"Backed up best model to Google Drive: {self.gdrive_best_model_path}")
                        
                        # Save training history alongside best model
                        self._save_training_history_to_gdrive(checkpoint_data)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to backup best model to Google Drive: {e}")
            
            # Backup periodic checkpoints to Google Drive (every 10 epochs or if specified)
            if (self.gdrive_backup and backup_to_gdrive and 
                (epoch % 10 == 0 or is_best or epoch < 5)):
                try:
                    gdrive_checkpoint_path = self.gdrive_dir / "models" / filename
                    shutil.copy2(checkpoint_path, gdrive_checkpoint_path)
                    self.logger.info(f"Backed up checkpoint to Google Drive: {gdrive_checkpoint_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to backup checkpoint to Google Drive: {e}")
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
            return str(checkpoint_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint {checkpoint_path}: {e}")
            raise e
    
    def load_checkpoint(self, checkpoint_path: str, fallback_to_gdrive: bool = True) -> Dict[str, Any]:
        """
        Load checkpoint from disk, with fallback to Google Drive.
        
        Args:
            checkpoint_path: Path to checkpoint file
            fallback_to_gdrive: Whether to try Google Drive if local file not found
            
        Returns:
            Dictionary containing model state and metadata
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Try local file first
        if checkpoint_path.exists():
            return self._load_checkpoint_file(checkpoint_path)
        
        # Try Google Drive fallback
        if fallback_to_gdrive and self.gdrive_backup:
            gdrive_path = self.gdrive_dir / "models" / checkpoint_path.name
            if gdrive_path.exists():
                self.logger.info(f"Loading checkpoint from Google Drive: {gdrive_path}")
                return self._load_checkpoint_file(gdrive_path)
        
        raise FileNotFoundError(f"Checkpoint not found locally or in Google Drive: {checkpoint_path}")
    
    def _load_checkpoint_file(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Load checkpoint from a specific file path."""
        try:
            # Load checkpoint (weights_only=False for compatibility with config objects)
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            self.logger.info(f"Loaded checkpoint: {checkpoint_path}")
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            raise e
    
    def load_best_model(self, fallback_to_gdrive: bool = True) -> Optional[Dict[str, Any]]:
        """
        Load the best saved model, with fallback to Google Drive.
        
        Args:
            fallback_to_gdrive: Whether to try Google Drive if local file not found
            
        Returns:
            Dictionary containing best model state, or None if not found
        """
        # Try local best model first
        if self.best_model_path.exists():
            return self._load_checkpoint_file(self.best_model_path)
        
        # Try Google Drive fallback
        if fallback_to_gdrive and self.gdrive_backup and self.gdrive_best_model_path:
            if self.gdrive_best_model_path.exists():
                self.logger.info(f"Loading best model from Google Drive: {self.gdrive_best_model_path}")
                return self._load_checkpoint_file(self.gdrive_best_model_path)
        
        self.logger.warning("Best model not found locally or in Google Drive")
        return None
    
    def load_latest_checkpoint(self, fallback_to_gdrive: bool = True) -> Optional[Dict[str, Any]]:
        """
        Load the most recent checkpoint, with fallback to Google Drive.
        
        Args:
            fallback_to_gdrive: Whether to try Google Drive if no local checkpoints
            
        Returns:
            Dictionary containing latest checkpoint state, or None if not found
        """
        # Try local checkpoints first
        checkpoint_files = self.list_checkpoints()
        
        if checkpoint_files:
            # Get most recent checkpoint (highest epoch number)
            latest_checkpoint = max(checkpoint_files, key=lambda x: x['epoch'])
            return self.load_checkpoint(latest_checkpoint['path'], fallback_to_gdrive=False)
        
        # Try Google Drive fallback
        if fallback_to_gdrive and self.gdrive_backup:
            gdrive_checkpoints = self.list_gdrive_checkpoints()
            if gdrive_checkpoints:
                latest_checkpoint = max(gdrive_checkpoints, key=lambda x: x['epoch'])
                self.logger.info(f"Loading latest checkpoint from Google Drive: {latest_checkpoint['path']}")
                return self._load_checkpoint_file(Path(latest_checkpoint['path']))
        
        self.logger.info("No checkpoints found locally or in Google Drive")
        return None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available local checkpoints.
        
        Returns:
            List of dictionaries with checkpoint information
        """
        return self._list_checkpoints_in_dir(self.checkpoint_dir)
    
    def list_gdrive_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available Google Drive checkpoints.
        
        Returns:
            List of dictionaries with checkpoint information
        """
        if not self.gdrive_backup or not self.gdrive_dir:
            return []
        
        gdrive_models_dir = self.gdrive_dir / "models"
        if not gdrive_models_dir.exists():
            return []
        
        return self._list_checkpoints_in_dir(gdrive_models_dir)
    
    def _list_checkpoints_in_dir(self, directory: Path) -> List[Dict[str, Any]]:
        """List checkpoints in a specific directory."""
        checkpoint_pattern = directory / "checkpoint_epoch_*.pth"
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
    
    def _save_training_history_to_gdrive(self, checkpoint_data: Dict[str, Any]) -> None:
        """Save training history and metrics to Google Drive."""
        if not self.gdrive_backup:
            return
        
        try:
            # Extract training history
            history = checkpoint_data.get('training_history', {})
            metrics = checkpoint_data.get('metrics', {})
            
            # Create summary data
            summary = {
                'epoch': checkpoint_data.get('epoch', 0),
                'best_cer': checkpoint_data.get('best_cer', 0.0),
                'training_history': history,
                'latest_metrics': metrics,
                'timestamp': checkpoint_data.get('timestamp'),
                'global_step': checkpoint_data.get('global_step', 0)
            }
            
            # Save to Google Drive
            history_path = self.gdrive_dir / "models" / "training_history.json"
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved training history to Google Drive: {history_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save training history to Google Drive: {e}")
    
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
    
    def sync_from_gdrive(self) -> bool:
        """
        Sync best model and latest checkpoint from Google Drive to local.
        
        Returns:
            True if any files were synced, False otherwise
        """
        if not self.gdrive_backup:
            self.logger.warning("Google Drive backup not enabled")
            return False
        
        synced = False
        
        try:
            # Sync best model
            if self.gdrive_best_model_path and self.gdrive_best_model_path.exists():
                if not self.best_model_path.exists():
                    shutil.copy2(self.gdrive_best_model_path, self.best_model_path)
                    self.logger.info(f"Synced best model from Google Drive")
                    synced = True
            
            # Sync latest checkpoint if no local checkpoints exist
            local_checkpoints = self.list_checkpoints()
            if not local_checkpoints:
                gdrive_checkpoints = self.list_gdrive_checkpoints()
                if gdrive_checkpoints:
                    latest_checkpoint = max(gdrive_checkpoints, key=lambda x: x['epoch'])
                    local_path = self.checkpoint_dir / Path(latest_checkpoint['path']).name
                    shutil.copy2(latest_checkpoint['path'], local_path)
                    self.logger.info(f"Synced latest checkpoint from Google Drive: {local_path}")
                    synced = True
            
            return synced
            
        except Exception as e:
            self.logger.error(f"Failed to sync from Google Drive: {e}")
            return False
    
    def get_gdrive_info(self) -> Dict[str, Any]:
        """
        Get information about Google Drive backup status.
        
        Returns:
            Dictionary with Google Drive information
        """
        if not self.gdrive_backup:
            return {"enabled": False}
        
        try:
            info = {
                "enabled": True,
                "gdrive_dir": str(self.gdrive_dir),
                "gdrive_accessible": self.gdrive_dir.exists() if self.gdrive_dir else False,
                "best_model_in_gdrive": self.gdrive_best_model_path.exists() if self.gdrive_best_model_path else False,
                "checkpoints_in_gdrive": len(self.list_gdrive_checkpoints())
            }
            
            # Calculate Google Drive usage
            if info["gdrive_accessible"]:
                gdrive_models_dir = self.gdrive_dir / "models"
                if gdrive_models_dir.exists():
                    total_size = 0
                    for file_path in gdrive_models_dir.glob("*.pth"):
                        total_size += os.path.getsize(file_path)
                    info["gdrive_usage_mb"] = total_size / (1024 * 1024)
                else:
                    info["gdrive_usage_mb"] = 0
            
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to get Google Drive info: {e}")
            return {"enabled": True, "error": str(e)} 