# encoding: utf-8

"""
Execute the Inference class on the two_visit_oct approach.
"""

# from two_visit_oct.models.OCT.LinearProjection import Model1
from model import LinearProjection

from dataset import NinetyDaysLatentMatching

from torch_device import get_device
import pandas as pd
import torch
import csv
import os


if __name__ == '__main__':
    hp = {
        'tensor_path': os.path.join('..', 'data_1', 'data_task1_fe', 'val'),
        'csv_path': os.path.join('..', 'data_2', 'df_task2_val_challenge.csv'),
        'save_path': os.path.join('..', 'results', '90_days_latent_matching_results.csv'),

        'batch_size': 1,
        'n_workers': 0,
        'device': get_device(),
        'dataset': NinetyDaysLatentMatching,

        # 90 days latent matching model
        'ninety_in_features': 1024,
        'ninety_out_features': 1024,
        'ninety_hidden_features': 1024,
        'model_path_latent_matching': os.path.join('..', 'ckpts', '90DaysLatentMatching', '14_09_2024-13_46_34', 'epoch_2.pth'),
        'model_latent_matching': LinearProjection,

        # task1 prediction model
        'task1_in_features': 2 * 1024,
        'task1_out_features': 4,
        'task1_hidden_features': 1000,
        'task1_model_path': os.path.join('..', 'ckpts', 'task2', 'disease_progression.pth'),
        'task1_model': LinearProjection,
    }

    val_dataset = pd.read_csv(hp['csv_path'])

    latent_matching_model = hp['model_latent_matching'](
        in_features=hp['ninety_in_features'],
        hidden_features=hp['ninety_hidden_features'],
        out_features=hp['ninety_out_features'],
        bias=True
    )
    latent_matching_model.load_state_dict(
        torch.load(
            hp['model_path_latent_matching'],
            map_location=hp['device'] if hp['device'] == torch.device('cuda') else torch.device('cpu')
        )['model']
    )
    latent_matching_model.to(hp['device'])
    latent_matching_model.eval()

    task1_model = hp['task1_model'](
        in_features=hp['task1_in_features'],
        hidden_features=hp['task1_hidden_features'],
        out_features=hp['task1_out_features'],
        bias=True
    )
    task1_model.load_state_dict(
        torch.load(
            hp['task1_model_path'],
            map_location=hp['device'] if hp['device'] == torch.device('cuda') else torch.device('cpu')
        )['model']
    )
    # task1_model = hp['task1_model'](4)
    # task1_model.load_state_dict(torch.load(hp['task1_model_path']))
    # task1_model.load_state_dict(torch.load(hp['task1_model_path'], map_location=hp['device'] if hp['device'] == torch.device('cuda') else torch.device('cpu')))
    task1_model.to(hp['device'])
    task1_model.eval()

    results = {}

    with torch.no_grad():
        for sample_idx in range(len(val_dataset)):
            sample = val_dataset.loc[sample_idx]

            x0 = torch.load(
                os.path.join(hp['tensor_path'], sample['image'][:-4] + '.pt'),
                map_location=hp['device']
            )
            x0 = x0.to(hp['device'])

            x90 = latent_matching_model(x0)

            pred = task1_model(torch.cat([x0, x90], dim=0))
            pred = torch.argmax(pred, dim=0).item()

            if pred == 3:
                pred = 1

            results[sample['case']] = pred

    with open(hp['save_path'], 'w') as csv_file:

        writer = csv.writer(csv_file, lineterminator='\n')
        writer.writerow(['case', 'prediction'])

        for key, value in results.items():
            writer.writerow([key, value])
