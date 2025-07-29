import matplotlib

from src.database import get_image
from src.roi import PolygonROISelector

matplotlib.use("Qt5Agg")

image_id = 34993
camera_name = "PPCX_Tele"
output_file = "data/PPCX_Tele_glacier_ROI.json"

img = get_image(image_id, camera_name=camera_name)
selector = PolygonROISelector(
    img,
    title="Select unstable area for DIC analysis",
    file_path=output_file,
)


print("Done.")
