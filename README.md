# torch_qat

## Requrements
pytorch=1.6.0 vision=0.7.0

## Dataset
`ImageNet`
`data_path` should include two folders(train and val). 
```python
dataset = torchvision.datasets.ImageFolder(
    os.path.join(data_path, "train"),
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ]))
dataset_test = torchvision.datasets.ImageFolder(
    os.path.join(data_path, "val"),
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
            ]))
```

## Network
`MobileNet-V2`

## Precision and Performance

This script supports float/ptq/qat.
All model pass 50000 ImageNet test.
| precision        | accuracy | size         | performance |
|------------------|----------|--------------|-------------|
| float            | 71.86    | 13.999657(MB)| |
| ptq(per-tensor)  | 56.39    | 3.631847(MB) | |
| ptq(per-channel) | 68.05    | (MB) | |
| qat              |          | (MB) | |

## Links

