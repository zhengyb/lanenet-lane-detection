
import numpy as np
from collections import OrderedDict
import time


# Lane cache an ordered dictionary of lanes, key is the lane id, value is the a dictionary of lane info
# {
#     "lane_id": {
#         "queue": [
#             {
#                 "frame_id": int,
#                 "fit_param": list,
#                 "x_range": list, # [x_min, x_max]
#                 "speed": list, # [linear_speed, angular_speed]
#                 "type": str, # "model" or "complete"
#             }
#         ],
#         "complete_lane"
#         "last_updated": int, # frame id of the last updated lane
#     }
# }


class SimpleLaneTracker:
    def __init__(self,
                 lane_cache_size=30,
                 id_queue_size=5):
        self.lane_cache_size = lane_cache_size
        self.id_queue_size = id_queue_size
        self.lanes_dict = OrderedDict()

    def track_lanes(self, image):
        pass
