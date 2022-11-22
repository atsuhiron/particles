import os
import shutil


class PathService:
    def __init__(self, frame_dir: str, frame_name: str = "frame", frame_num_order: int = 5):
        self.frame_dir = frame_dir
        self.frame_name = frame_name
        self.frame_num_order = frame_num_order
        
    def gen_frame_path(self, index: int) -> str:
        number = str(index).zfill(self.frame_num_order)
        return "{}/{}_{}.png".format(self.frame_dir, self.frame_name, number)

    def gen_template_frame_path(self) -> str:
        templ = "%0{}d".format(self.frame_num_order)
        return "{}/{}_{}.png".format(self.frame_dir, self.frame_name, templ)

    def reset_dir(self):
        if os.path.exists(self.frame_dir):
            shutil.rmtree(self.frame_dir)
        os.makedirs(self.frame_dir)
