import sys
import types
import os

# 1. Patch flop_counter
if 'torch.utils.flop_counter' not in sys.modules:
    mod = types.ModuleType('torch.utils.flop_counter')
    mod.FlopCounterMode = object
    sys.modules['torch.utils.flop_counter'] = mod

# 2. Patch torch.compiler
import torch
if not hasattr(torch, "compiler"):
    class DummyCompiler:
        @staticmethod
        def disable(func):
            return func
    torch.compiler = DummyCompiler()
    sys.modules['torch.compiler'] = torch.compiler

# 3. Patch swa_utils
import torch.optim.swa_utils
if not hasattr(torch.optim.swa_utils, "get_ema_avg_fn"):
    print("Patching get_ema_avg_fn")
    def get_ema_avg_fn(decay=0.999):
        return lambda averaged_param, model_param, num_averaged: None
    torch.optim.swa_utils.get_ema_avg_fn = get_ema_avg_fn

print("swa_utils has get_ema_avg_fn:", hasattr(torch.optim.swa_utils, "get_ema_avg_fn"))

# Now try importing pytorch_lightning's failing line
try:
    from pytorch_lightning.callbacks.weight_averaging import EMAWeightAveraging
    print("Successfully imported lightning callback!")
except ImportError as e:
    import traceback
    print("Failed lightning import:")
    traceback.print_exc()
