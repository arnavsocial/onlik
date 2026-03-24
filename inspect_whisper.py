import sys
import types
import os
import inspect

import faster_whisper.transcribe

print("Inspection for TranscriptionOptions.__init__:")
try:
    spec = inspect.getfullargspec(faster_whisper.transcribe.TranscriptionOptions.__init__)
    print("Args:", spec.args)
    print("Defaults:", spec.defaults)
    print("Kwonlyargs:", spec.kwonlyargs)
    print("Kwonlydefaults:", spec.kwonlydefaults)
except Exception as e:
    print("Failed to inspect:", e)
