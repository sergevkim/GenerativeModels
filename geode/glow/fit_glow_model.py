from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi

import argparse

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

import wandb
import torchvision

from geode.glow import Glow
from my_utils import set_seed, count_parameters, calc_z_shapes
from my_calculate_fid import calculate_fid
from inception import InceptionV3


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Glow trainer")
parser.add_argument("--batch", default=64, type=int, help="batch size")
parser.add_argument("--iter", default=200000, type=int, help="maximum iterations")
parser.add_argument("--num_epochs", default=30, type=int, help="num epochs")
parser.add_argument(
    "--n_flow", default=16, type=int, help="number of flows in each block"
)
parser.add_argument("--n_block", default=3, type=int, help="number of blocks")
parser.add_argument(
    "--no_lu",
    action="store_true",
    help="use plain convolution instead of LU decomposed version",
)
parser.add_argument(
    "--affine", action="store_true", help="use affine coupling instead of additive"
)
parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
parser.add_argument("--lr", default=2e-4, type=float, help="learning rate")
parser.add_argument("--img_size", default=64, type=int, help="image size")
parser.add_argument("--temp", default=0.5, type=float, help="temperature of sampling")
parser.add_argument("--n_sample", default=20, type=int, help="number of samples")



def wandb_start(config, run_name):
    wandb.init(project="dgm-ht4", config=config)
    wandb.run.name = run_name


def get_loaders(BS, image_size):
    celeba_transforms = transforms.Compose([
        transforms.CenterCrop((148, 148)),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    my_dataset = torchvision.datasets.CelebA('./celeba/',
                                           transform=celeba_transforms,
                                           download=False)
    all_size = len(my_dataset)
    
    val_size = 2000
    train_size = all_size - val_size
    train_set, val_set = torch.utils.data.random_split(my_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=BS, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)

    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    fid_loader = DataLoader(val_set, batch_size=BS, shuffle=False, drop_last=True)
    return train_loader, val_loader, fid_loader


def calc_loss(log_p, logdet, image_size, n_bins):
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * 3

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )




def train(args, model, optimizer, classifier):
    train_loader, val_loader, fid_loader = get_loaders(args.batch, args.img_size)
    n_bins = 2.0 ** args.n_bits

    for ep_num in range(args.num_epochs): 
        model.train()
        for index, image_attr in tqdm(enumerate(train_loader)): ################
            image = image_attr[0].to("cuda")

            image = image * 255

            if args.n_bits < 8:
                image = torch.floor(image / 2 ** (8 - args.n_bits))

            image = image / n_bins - 0.5

            if index == 0:
                with torch.no_grad():
                    log_p, logdet, _ = model(
                        image + torch.rand_like(image) / n_bins
                    )

                    continue

            else:
                log_p, logdet, _ = model(image + torch.rand_like(image) / n_bins)

            logdet = logdet.mean()

            loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
            model.zero_grad()
            loss.backward()
            # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
            #warmup_lr = args.lr
            #optimizer.param_groups[0]["lr"] = warmup_lr
            optimizer.step()

            # tracking
            wandb.log({'Loss':loss.item(),
                       'logP':log_p.item(),
                       'logdet':log_det.item(),
                      })

        ########
        #z_sample = []
        #z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
        #for z in z_shapes:
        #    z_new = torch.randn(args.n_sample, *z) * args.temp
        #    z_sample.append(z_new.to(device))

        #sampled_image = model.reverse(z_sample)
        #print(sampled_image)
        #break
        #######
        torch.save(model.state_dict(), 'model_' + str(ep_num))
        
        with torch.no_grad():
            
            fid = calculate_fid(args, fid_loader, model, classifier)
            wandb.log({'FID':fid})


            # deprecation
            model.eval()

            for ind, image_attr in enumerate(val_loader):  # batch = 1
                if ind >= 10: break
                image = image_attr[0].to('cuda')   # batch=1

                _, _, z = model(image)

                fake_image = model.reverse(z).detach().cpu()[0]
                image = image.detach().cpu()[0]


                z_sample = []
                z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
                for z in z_shapes:
                    z_new = torch.randn(args.n_sample, *z) * args.temp
                    z_sample.append(z_new.to('cuda'))

                sampled_image = model.reverse(z_sample).detach().cpu()[0]

                wandb.log({"samples": 
                           [wandb.Image(image.permute(1, 2, 0).numpy(), 
                                        caption='real'),
                            wandb.Image(fake_image.permute(1, 2, 0).numpy(), 
                                        caption='fake')]
                          ,
                           "GENERATED":
                           [wandb.Image(sampled_image.permute(1, 2, 0).numpy(), 
                            caption='generated from noise')]
                           })

        model.train()


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    set_seed(21)
    
    wandb.login(key='6aa2251ef1ea5e572e6a7608c0152db29bd9a294')
    wandb_start(args, 'GLOW-128-3e-4')

    model = Glow(
        3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    )
    model = model.to(device)
    print(count_parameters(model))
    wandb.watch(model)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    
    # FID
    classifier = InceptionV3()
    classifier.to(device)
    
    train(args, model, optimizer, classifier)
