import time
import numpy as np

# Isolate: is float32->int8 conversion inside execute_async the cost,
# or is it something else? Test by pre-casting to int8 ourselves.
canvas = np.random.randint(0, 255, (640,640,3), dtype=np.uint8)
t0 = time.perf_counter()
x_float = canvas.astype(np.float32)[np.newaxis,...]
t1 = time.perf_counter()
print(f"float32 cast only: {(t1-t0)*1000:.3f} ms")
