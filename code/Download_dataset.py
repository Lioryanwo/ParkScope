# Downloading the 'Roadside Parking' dataset from Roboflow via API.
from roboflow import Roboflow
rf = Roboflow(api_key="wQZahQEnO6FVBKvlZq40")
project = rf.workspace("none-dhjtb").project("roadside-parking-slots-2dpia")
version = project.version(1)
dataset = version.download("yolov11")
                