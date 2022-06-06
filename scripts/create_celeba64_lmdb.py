# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import lmdb
import os
import torchvision.datasets as dset
import torchvision
import torch
import torchvision.transforms as transforms

res_size = 224

def main(split, img_path, lmdb_path):
    assert split in {"train", "valid", "test"}
    # create target directory
    if not os.path.exists(lmdb_path):
        os.makedirs(lmdb_path, exist_ok=True)

    lmdb_split = {'train': 'train', 'valid': 'validation', 'test': 'test'}[split]
    lmdb_path = os.path.join(lmdb_path, '%s.lmdb' % lmdb_split)

    # if you don't have this will download the data
    #data = dset.celeba.CelebA(root=img_path, split=split, target_type='attr', transform=None, download=True)
    print(len('total data'))


    train_transform = transforms.Compose([transforms.Resize([int(res_size*1.15), int(res_size*1.15)]),
                                transforms.RandomCrop(size=res_size),
                                transforms.RandomHorizontalFlip(),
                                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                transforms.ToTensor()])

    data = torchvision.datasets.ImageFolder(img_path, transform=train_transform)
    loader = torch.utils.data.DataLoader(data, batch_size=1, num_workers=6)

    # create lmdb
    env = lmdb.open(lmdb_path, map_size=1e12)
    with env.begin(write=True) as txn:
        #for i in range(len(loader.dataset)):
        for i in data.imgs:
            #file_path = os.path.join(data.root, data.base_folder, "img_align_celeba", data.filename[i])
            file_path = i[0]
            #attr = data.attr[i, :]
            with open(file_path, 'rb') as f:
                file_data = f.read()

            txn.put(str(i).encode(), file_data)
            print(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CelebA 64 LMDB creator.')
    # experimental results
    parser.add_argument('--img_path', type=str, default='/data1/datasets/celeba_org/',
                        help='location of images for CelebA dataset')
    parser.add_argument('--lmdb_path', type=str, default='/data1/datasets/celeba_org/celeba64_lmdb',
                        help='target location for storing lmdb files')
    parser.add_argument('--split', type=str, default='train',
                        help='training or validation split', choices=["train", "valid", "test"])
    args = parser.parse_args()
    main(args.split, args.img_path, args.lmdb_path)

