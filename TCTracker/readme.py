try: from data.dataset import *
except: from TCTracker.data.dataset import *

try: from evaluation.IOU import *
except: from TCTracker.evaluation.IOU import *

try: from evaluation.Metrics import *
except: from TCTracker.evaluation.Metrics import *

try: from pipeline.classifier import *
except: from TCTracker.pipeline.classifier import *

try: from pipeline.detector import *
except: from TCTracker.pipeline.detector import *

try: from pipeline.TCTracker import *
except: from TCTracker.pipeline.TCTracker import *

__all__ = [k for k in globals().keys() if not k.startswith("_")]
