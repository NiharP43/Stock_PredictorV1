import numpy as np, pandas as pd
import yfinance as yf
from transformers import pipeline
import torch, sklearn
print("Setup OK:\n",
      "  numpy", np.__version__,
      "  pandas", pd.__version__,
      "  yfinance", yf.__version__,
      "  transformers", pipeline,
      "  torch", torch.__version__,
      "  sklearn", sklearn.__version__)
