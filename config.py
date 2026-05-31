
ARROW_FILE = "NR_WebDataset/data-00000-of-00001.arrow"
OUTPUT_CSV = "results2.csv"
BASE_URL = "https://ollama-gpt-oss.cluster.ai.wu.ac.at/"
MODEL = "gemma4:latest"
LIMIT = None        # set to an int to cap the number of pairs processed
BATCH_SIZE = 10
DELAY_SECONDS = 0.5
MAX_RETRIES = 1

# "A" — binary, "B" — two-class, "C" — three-class
STRATEGIES_TO_RUN = ["A"]
