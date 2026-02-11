import logging
from pathlib import Path
from shrekommender_system.data.ingestion import KafkaDataIngester

logger = logging.getLogger(__name__)


class Paths:
    def __init__(self, data_root: Path, model_dir: Path):
        self.data_root = data_root
        self.model_dir = model_dir

    def ensure_data_dirs(self, *subdirs):
        for subdir in subdirs:
            (self.data_root / subdir).mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)


class Pipeline:
    #Main pipeline connecting ingestion, preprocessing, and model updating

    def __init__(self):
        self.ingester = KafkaDataIngester()
        self.paths = Paths(Path("."), Path("./models"))
        self.model = None

    def run(self, n: int = 10):
        stats = {"ingestion": None, "preprocessing": None, "model": None}

        # Ensure directories exist
        self.paths.ensure_data_dirs("raw", "processed", "intermediate")

        # Ingestion
        try:
            ingestion_result = self.ingester.fetch_recent(n=n)
            stats["ingestion"] = ingestion_result
            logger.info("Ingestion completed.")
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            stats["ingestion"] = {"status": "failed", "error": str(e)}
            return stats

        # Preprocessing (placeholder)
        try:
            stats["preprocessing"] = {"status": "completed", "records_cleaned": 10}
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            stats["preprocessing"] = {"status": "failed", "error": str(e)}
            return stats

        # Model update (placeholder)
        try:
            model_path = self.paths.model_dir / "latest.pkl"
            import pickle
            with open(model_path, "wb") as f:
                pickle.dump({"model": "dummy", "most_popular_movies": [0, 1, 2]}, f)
            stats["model"] = {"status": "updated", "model_path": str(model_path)}
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Model update failed: {e}")
            stats["model"] = {"status": "failed", "error": str(e)}

        return stats
