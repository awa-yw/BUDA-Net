import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class GoProDataset(Dataset):
    """
    GoPro dataset with structure:
    root/gopro
      ├── blur/image/*.png
      └── sharp/image/*.png

    Train / test split is done by index.
    """
    def __init__(
        self,
        root,
        split="train",
        crop_size=256,
        training=True,
        train_ratio=0.8,
        extensions=(".png", ".jpg", ".jpeg")
    ):
        super().__init__()

        self.blur_dir = os.path.join(root, "blur", "images")
        self.sharp_dir = os.path.join(root, "sharp", "images")

        assert os.path.isdir(self.blur_dir), f"Not found: {self.blur_dir}"
        assert os.path.isdir(self.sharp_dir), f"Not found: {self.sharp_dir}"

        names = [
            f for f in os.listdir(self.blur_dir)
            if f.lower().endswith(extensions)
        ]
        names.sort()

        # ---- train / test split ----
        split_idx = int(len(names) * train_ratio)
        if split == "train":
            self.names = names[:split_idx]
        elif split == "test":
            self.names = names[split_idx:]
        else:
            raise ValueError("split must be 'train' or 'test'")

        self.crop_size = crop_size
        self.training = training

        print(
            f"[GoProDataset] split={split}, "
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
