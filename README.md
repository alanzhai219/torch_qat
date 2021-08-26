# torch_qat

## Requrements
pytorch=1.6.0 vision=0.7.0

## Dataset
`ImageNet`
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