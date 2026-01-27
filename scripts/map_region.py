import matplotlib

from ppcluster._deprecated.roi import PolygonROISelector
from ppcluster.utils.database import get_image

matplotlib.use("Qt5Agg")

image_id = 34993
camera_name = "PPCX_Tele"
output_file = "data/PPCX_Tele_glacier_ROI.json"

img = get_image(
    image_id, camera_name=camera_name, app_host="150.145.51.193", app_port="8080"
)
selector = PolygonROISelector(
    file_path=output_file,
)
selector.draw_interactive(img)

print("Done.")
