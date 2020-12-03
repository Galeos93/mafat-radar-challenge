import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torchvision.utils import make_grid

from mafat_radar_challenge.base import TrainerBase, AverageMeter
from mafat_radar_challenge.utils import setup_logger


log = setup_logger(__name__)


class MAFATTester(TrainerBase):
    """
    Responsible for testing SIIM Test dataset
    """

    def __init__(self, config, device, data_loader):
        self.cfg = config
        self.device = device

    def create_submission(self, model, data_loader):
        preds = list()
        output_df = pd.DataFrame({"segment_id": data_loader.dataset.df.segment_id})
        with torch.no_grad():
            for i, data in enumerate(tqdm(data_loader)):
                if isinstance(data, list):
                    for i, _ in enumerate(data):
                        data[i] = data[i].to(self.device)
                else:
                    data = data.to(self.device)
                output = model(data).cpu()
                output = torch.sigmoid(output)
                preds.append(output.cpu().numpy())
        preds = np.vstack(preds)
        output_df["prediction"] = preds.reshape(-1)
        return output_df
