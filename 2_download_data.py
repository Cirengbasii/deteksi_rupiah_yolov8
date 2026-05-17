from roboflow import Roboflow
rf = Roboflow(api_key="hjcrvaoQmdqxm47kKmSu")
project = rf.workspace("fix-tmc9q").project("uang-2016-2022")
version = project.version(1)
dataset = version.download("yolov8")
                