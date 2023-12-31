import logging
import lancedb
import os
from pathlib import Path


DB_TABLE_NAME = "split_files_db"

# Setting up the logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# db
db_uri = os.path.join(Path(__file__).parents[1], ".lancedb")
db = lancedb.connect(db_uri)
table = db.open_table(DB_TABLE_NAME)
