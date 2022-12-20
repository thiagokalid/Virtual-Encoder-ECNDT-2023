import os


media = ["air", "water"]
excurtion_types = ["calibration_y", "closed_loop", "single_x", "single_y"]
for med in media:
    for ex in excurtion_types:
        data_root = f"../../media/planar/{med}/"
        extension = ".jpg"

        path = data_root + ex + "_video"
        list_of_files = os.listdir(data_root + ex + "_video")
        list_of_files = [int(file.replace(extension, "").replace("image", "")) for file in list_of_files]
        list_of_files.sort()

        for i in range(len(list_of_files)):
            original_idx = list_of_files[i]
            new_idx = i + 1
            original_name = f"image{original_idx:02d}" + extension
            new_name = f"image{new_idx:02d}" + extension
            os.rename(path + "/" + original_name, path + "/" + new_name)
