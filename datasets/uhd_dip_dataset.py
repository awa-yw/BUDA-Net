import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class UHDDIPDataset(Dataset):
    """
    UHD-DIP dataset structure (your version):

    root/
      ├── train/
      │   ├── input_new/   (blur)
      │   └── gt_new/      (sharp)
      └── test/
          ├── input300/   (blur)
          └── gt300/      (sharp)
    """
    def __init__(
        self,
        root,
        split="train",
        crop_size=256,
        training=True,
        extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif")
    ):
        super().__init__()

        if split == "train":
            self.blur_dir = os.path.join(root, "train", "input_new")
            self.sharp_dir = os.path.join(root, "train", "gt_new")
        elif split == "test":
            self.blur_dir = os.path.join(root, "test", "input300")
            self.sharp_dir = os.path.join(root, "test", "gt300")
        else:
            raise ValueError("split must be 'train' or 'test'")

        assert os.path.isdir(self.blur_dir), f"Not found: {self.blur_dir}"
        assert os.path.isdir(self.sharp_dir), f"Not found: {self.sharp_dir}"

        # ---- filter valid images ----
        def valid(name):
            name_l = name.lower()
            return (
                (name_l.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")))
                and not name_l.startswith(".")
                and "baiduyun" not in name_l
            )


        self.names = sorted(
            f for f in os.listdir(self.blur_dir)
            if valid(f)
        )

        self.crop_size = crop_size
        self.training = training

        print(
            f"[UHDDIPDataset] split={split}, "
            f"samples={len(self.names)}"
        )

    def __len__(self):
        return len(self.names)

    def _random_crop(self, img1, img2):
        w, h = img1.size
        cs = self.crop_size

        if w < cs or h < cs:
            img1 = TF.resize(img1, (cs, cs))
            img2 = TF.resize(img2, (cs, cs))
            return img1, img2

        x = random.randint(0, w - cs)
        y = random.randint(0, h - cs)
        img1 = TF.crop(img1, y, x, cs, cs)
        img2 = TF.crop(img2, y, x, cs, cs)
        return img1, img2

    def __getitem__(self, idx):
        name = self.names[idx]

        blur = Image.open(os.path.join(self.blur_dir, name)).convert("RGB")
        sharp = Image.open(os.path.join(self.sharp_dir, name)).convert("RGB")

        if self.training:
            blur, sharp = self._random_crop(blur, sharp)

            if random.random() < 0.5:
                blur = TF.hflip(blur)
                sharp = TF.hflip(sharp)
            if random.random() < 0.5:
                blur = TF.vflip(blur)
                sharp = TF.vflip(sharp)

        blur = TF.to_tensor(blur)
        sharp = TF.to_tensor(sharp)
        return blur, sharp
