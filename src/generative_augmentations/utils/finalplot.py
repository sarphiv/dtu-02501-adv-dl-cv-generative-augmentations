from pathlib import Path
import re

import torch as th
import numpy as np
import matplotlib.pyplot as plt
import tyro
from src.generative_augmentations.config import Config
from src.generative_augmentations.utils.metrics import compute_final_iou
from src.generative_augmentations.models.deeplab import DeepLabv3Lightning
from src.generative_augmentations.datasets.datamodule import COCODataModule
from src.generative_augmentations.datasets.transforms import transforms
from src.generative_augmentations.datasets.coco import index_to_name



def iou_from_checkpoint(checkpoint, file_name, cfg):
   
    datamodule = COCODataModule(data_dir=Path(cfg.dataloader.data_dir), transform_val=transforms["final transform"], val_data_path=Path('test'))
    val_loader = datamodule.val_dataloader()

    model = DeepLabv3Lightning.load_from_checkpoint(checkpoint)
    model.eval()
    model = model.to('cuda')

    IoU, Acc, mIoU, mAcc = compute_final_iou(model, val_loader)

    stats = {"IoU": IoU, "Acc": Acc, "mIoU": mIoU, "mAcc": mAcc}
    
    th.save(stats, file_name)

def get_dictionary(file_names, folder): 
    data_dict = {} 
    for file_name in file_names: 
        data = th.load(folder / file_name)
        trans = file_name[-5:-3] # file_name[-2:]
        aug = file_name[:2]
        data_amount = file_name[3:-6]# file_name[3:-3]
        if aug == 'no': 
            if trans == 'no': 
                name = 'no-transform'
            elif trans == 'si': 
                name = 'vanilla-transform'
            elif trans == 'ad': 
                name = 'advanced-transform'
        if aug == 'in': 
            if trans == 'si': 
                name = 'vanilla-instance'
            elif trans == 'ad': 
                name = 'advanced-instance'
        if aug == 'di': 
            name = 'vanilla-diffusion'
        
        if not name in data_dict.keys(): 
            data_dict[name] = {}
        data_dict[name][data_amount] = data 

    return data_dict

def plot_data(data, metric, figure_name, title, id=None): 
    plt.figure(figsize=(8, 5))

    for model_name, results in data.items():
        # Sort x (fractions) to ensure proper line plotting
        idx = sorted(results.keys())
        x = np.array([float(k) for k in idx]) / 10
        if id == None: 
            y = [results[f][metric] for f in idx]
        else : 
            y = [results[f][metric][id] for f in idx]
        plt.plot(x, y, marker='o', label=model_name)

    plt.xlabel("Fraction of COCO")
    plt.xticks(x, labels=[f"{f*100:.0f}%" for f in x])
    plt.ylabel(f"{metric}")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figure_name)
    plt.clf()
    # plt.show()


if __name__ == "__main__": 
    args = tyro.cli(Config)
    models_dir = Path('../scratch/models')
    model_names = [
        "no_1.0_no_happy-gorge-227",
        "no_0.5_no_likely-disco-228",
        "no_0.25_no_wandering-vortex-236",
        "no_0.125_no_youthful-oath-239",
        "no_1.0_si_zesty-jazz-248",
        "no_0.5_si_solar-wildflower-250",
        "no_0.25_si_proud-disco-251",
        "no_0.125_si_rosy-butterfly-252",
        "no_1.0_ad_dauntless-feather-229",
        "no_0.5_ad_vocal-sun-237",
        "no_0.25_ad_wise-vortex-246",
        "no_0.125_ad_gentle-frog-249",
        "in_1.0_si_elated-sunset-242",
        "in_0.5_si_quiet-tree-243",
        "in_0.25_si_flowing-meadow-244",
        "in_0.125_si_fine-breeze-245",
        "in_1.0_ad_ancient-serenity-247",
        "in_0.5_ad_resilient-spaceship-255",
        "in_0.25_ad_atomic-pond-258", # removed E
        "in_0.125_ad_glowing-rain-257", # removed E
        'di_0.125_si_atomic-serenity-240',
        'di_0.25_si_vague-silence-238',
        'di_0.5_si_skilled-rain-235',
        'di_1.0_si_faithful-frost-234'
    ]
    
    get_prefix = lambda s: re.match(r"^(.*)_", s)
    run_params = []  
    for ckpt in model_names: 
        # ckpt_path = models_dir / ckpt / 'last.ckpt'
        exp_name = get_prefix(ckpt).group(1) + '.pt'
        run_params.append(exp_name)
        # iou_from_checkpoint(ckpt_path, file_name=exp_name, cfg=args)
    
    data_dict = get_dictionary(run_params, folder=Path('final_metrics'))
    plot_data(data_dict, metric = "mIoU", figure_name="final_plots/IoU_all", title="mIoU for 20 % of validation COCO")
        

    # Plot all IoUs to see if there are outliers. 
    for key, value in index_to_name.items():
        data_dict = get_dictionary(run_params, folder=Path('final_metrics'))
        plot_data(data_dict, metric = "IoU", figure_name=f"final_plots/IoU_{value}", title=f"IoU for 20 % of validation COCO on {value}", id=key)
        
