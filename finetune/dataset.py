import pickle
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class ProgramGridDataset(Dataset):
    def __init__(self, grids_dir, programs_dir, tokenizer):
        self._grids_dir = Path(grids_dir)
        self._programs_dir = Path(programs_dir)
        self._tokenizer = tokenizer

        # load the data files in
        self._grids = np.load(self._grids_dir)
        self._programs = pickle.load(open(self._programs_dir, "rb"))

    def __len__(self):
        return len(self._programs)

    def __getitem__(self, idx):
        # preprocess the grid with the start point
        grid = self._grids[idx]

        # repeat the grid across three channels
        grid = np.repeat(grid[np.newaxis, :, :], 3, axis=0)
        return (
            torch.from_numpy(grid),
            self._tokenizer.encode(
                self._programs[idx],
                max_length=512,
                padding="max_length",
                return_tensors="pt",
            ),
        )


if __name__ == "__main__":
    p = ProgramGridDataset(
        grids_dir="gen/data/grids.npy",
        starts_dir="gen/data/pts.npy",
        programs_dir="gen/data/programs.json",
    )
    print(p[0])
