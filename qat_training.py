import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import os
import time
import sys
import torch.quantization

# Specify random seed for repeatable results
# torch.manual_seed(191009)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def evaluate(model, criterion, data_loader, neval_batches):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print('.', end = '')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt >= neval_batches:
                 return top1, top5

    return top1, top5

def load_model(model_file):
    model = MobileNetV2()
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to('cpu')
    return model

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def prepare_data_loaders(args):
    print("Loading Data Base ...")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dir = os.path.join(args.data_path, 'train')
    val_dir = os.path.join(args.data_path, 'val')
    dataset = torchvision.datasets.ImageFolder(
        train_dir ,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ]))
    dataset_test = torchvision.datasets.ImageFolder(
        val_dir,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
            ]))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    return data_loader, data_loader_test

data_path = '/data/pytorch/datasets/imagenet_training' # such folder should include train and val.
saved_model_dir = 'data/'
float_model_file = 'mobilenet_v2-b0353104.pth'#'mobilenet_pretrained_float.pth'
scripted_float_model_file = 'mobilenet_quantization_scripted.pth'
scripted_quantized_model_file = 'mobilenet_quantization_scripted_quantized.pth'

train_batch_size = 30
eval_batch_size = 50

data_loader, data_loader_test = prepare_data_loaders(data_path)
criterion = nn.CrossEntropyLoss()

'''
############################ QAT ###########################
'''
print("####### running QAT #######")
def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
    model.train()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')

    cnt = 0
    for image, target in data_loader:
        start_time = time.time()
        print('.', end = '')
        cnt += 1
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        avgloss.update(loss, image.size(0))
        if cnt >= ntrain_batches:
            print('Loss', avgloss.avg)

            print('Training: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
            return

    print('Full imagenet train set:  * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=top1, top5=top5))
    return


def run_benchmark(model_file, img_loader):
    '''
    ####################### benchmark ######################
    '''
    elapsed = 0
    model = torch.jit.load(model_file)
    model.eval()
    num_batches = 5
    # Run the scripted model on a few batches of images
    for i, (images, target) in enumerate(img_loader):
        if i < num_batches:
            start = time.time()
            output = model(images)
            end = time.time()
            elapsed = elapsed + (end-start)
        else:
            break
    num_images = images.size()[0] * num_batches

    print('Elapsed time: %3.0f ms' % (elapsed/num_images*1000))
    return elapsed

run_benchmark(saved_model_dir + scripted_float_model_file, data_loader_test)

run_benchmark(saved_model_dir + scripted_quantized_model_file, data_loader_test)

def run_fp32_baseline(args):
    '''
    ######################## baseline ##########################
    ###### 71.9% on the eval dataset of 50,000 images ######
    '''
    print("####### running baseline #######")
    float_model = load_model(saved_model_dir + float_model_file).to('cpu')
    
    # Next, we'll "fuse modules"; this can both make the model faster by saving on memory access
    # while also improving numerical accuracy. While this can be used with any model, this is
    # especially common with quantized models.
    
    print('\n Inverted Residual Block: Before fusion \n\n', float_model.features[1].conv)
    float_model.eval()
    
    # Fuses modules
    float_model.fuse_model()
    
    # Note fusion of Conv+BN+Relu and Conv+Relu
    print('\n Inverted Residual Block: After fusion\n\n',float_model.features[1].conv)
    
    num_calibration_batches = 32
    num_eval_batches = 1000
    
    print("Size of baseline model")
    print_size_of_model(float_model)
    
    top1, top5 = evaluate(float_model, criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
    torch.jit.save(torch.jit.script(float_model), saved_model_dir + scripted_float_model_file)

def run_ptq_per_tensor(args):
    '''
    ######################## PTQ(per-tensor)##########################
    accuracy of 56.7% on the eval dataset
    '''
    print("####### running PTQ(per-tensor) #######")
    num_calibration_batches = 32
    
    myModel = load_model(saved_model_dir + float_model_file).to('cpu')
    myModel.eval()
    
    # Fuse Conv, bn and relu
    myModel.fuse_model()
    
    # Specify quantization configuration
    # Start with simple min/max range estimation and per-tensor quantization of weights
    myModel.qconfig = torch.quantization.default_qconfig
    print(myModel.qconfig)
    torch.quantization.prepare(myModel, inplace=True)
    
    # Calibrate first
    print('Post Training Quantization Prepare: Inserting Observers')
    print('\n Inverted Residual Block:After observer insertion \n\n', myModel.features[1].conv)
    
    # Calibrate with the training set
    evaluate(myModel, criterion, data_loader, neval_batches=num_calibration_batches)
    print('Post Training Quantization(per-tensor): Calibration done')
    
    # Convert to quantized model
    torch.quantization.convert(myModel, inplace=True)
    print('Post Training Quantization: Convert done')
    print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n',myModel.features[1].conv)
    
    print("Size of model after quantization")
    print_size_of_model(myModel)
    
    top1, top5 = evaluate(myModel, criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))

def run_ptq_per_channel(args):
    '''
    ######################## PTQ(per-channel) ##########################
    accuracy to over 67.3% on the eval dataset
    '''
    print("####### running PTQ(per-channel) #######")
    per_channel_quantized_model = load_model(saved_model_dir + float_model_file)
    per_channel_quantized_model.eval()
    per_channel_quantized_model.fuse_model()
    per_channel_quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    print(per_channel_quantized_model.qconfig)
    
    torch.quantization.prepare(per_channel_quantized_model, inplace=True)
    evaluate(per_channel_quantized_model,criterion, data_loader, num_calibration_batches)
    print('Post Training Quantization(per-channel): Calibration done')
    
    torch.quantization.convert(per_channel_quantized_model, inplace=True)
    top1, top5 = evaluate(per_channel_quantized_model, criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
    torch.jit.save(torch.jit.script(per_channel_quantized_model), saved_model_dir + scripted_quantized_model_file)

def run_qat(args):
    qat_model = load_model(saved_model_dir + float_model_file)
    qat_model.fuse_model()
    
    optimizer = torch.optim.SGD(qat_model.parameters(), lr = 0.0001, momentum=0.9, gamma=0.1)
    qat_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    
    torch.quantization.prepare_qat(qat_model, inplace=True)
    print('Inverted Residual Block: After preparation for QAT, note fake-quantization modules \n',qat_model.features[1].conv)
    
    num_train_batches = 20
    
    # QAT takes time and one needs to train over a few epochs.
    # Train and check accuracy after each epoch
    for nepoch in range(8):
        train_one_epoch(qat_model, criterion, optimizer, data_loader, torch.device('cpu'), num_train_batches)
        if nepoch > 3:
            # Freeze quantizer parameters
            qat_model.apply(torch.quantization.disable_observer)
        if nepoch > 2:
            # Freeze batch norm mean and variance estimates
            qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    
        # Check the accuracy after each epoch
        quantized_model = torch.quantization.convert(qat_model.eval(), inplace=False)
        quantized_model.eval()
        top1, top5 = evaluate(quantized_model,criterion, data_loader_test, neval_batches=num_eval_batches)
        print('Epoch %d :Evaluation accuracy on %d images, %2.2f'%(nepoch, num_eval_batches * eval_batch_size, top1.avg))

def main(args):
    run_fp32_baseline(args)
    run_ptq_per_tensor(args)
    run_ptq_per_channel(args)
    run_qat(args)
    run_benchmark()

def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Quantized Test', add_help=add_help)
    parser.add_argument('--data-path',
                        default='/data/imagenet',
                        help='dataset')
    parser.add_argument('--workers',
                        default=16,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 16)')
    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)