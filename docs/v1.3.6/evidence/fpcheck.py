import time
from statexplore.core.importers import import_file
from statexplore.core import identity
d = import_file(r"E:\sx_v136_work\kit\fixtures\StatExplore_Test54B_100K.csv")
print(d.raw.shape, d.raw.dtypes.value_counts().to_dict())
identity._VALUES.clear()
t = time.perf_counter(); identity.values_digest(d.raw); print("cold values_digest s", round(time.perf_counter() - t, 4))
from statexplore.discovery.engine import data_signature
identity._VALUES.clear(); identity._REALIZED.clear()
t = time.perf_counter(); data_signature(d); print("cold data_signature s", round(time.perf_counter() - t, 4))
t = time.perf_counter(); [data_signature(d) for _ in range(100)]; print("warm per call ms", round((time.perf_counter() - t) * 10, 3))
