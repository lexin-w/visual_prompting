import shutil
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
from torch.utils.data import random_split
from PIL import Image
from torchvision import transforms

class ImgDataset(Dataset):
  def __init__(self, data, label_texts):
    self.classes = label_texts
    self.data = data
    self.length = len(data)

  def __getitem__(self, index):
    return self.data[index][0], self.data[index][1]
  
  def __len__(self):
    return self.length

def load_data(text_path, img_path):
    text_data = pd.read_csv(text_path) # changed
    text_data = text_data.dropna() # added
#     text_data.columns=['id','text_a','text_b','label']
    image_dirs = os.listdir(img_path)
    image_ids = [f[:-4] for f in image_dirs]
    text_data = text_data[text_data['id'].isin(image_ids)]
#     text_data['text'] = text_data['text_a']+" "+text_data['text_b'] # comment out
#     texts = list(text_data['text'])
#     clip_texts = [text[:155] for text in texts]
   
    text_labels = np.array(text_data['label'].unique())
#     labels = [0.0 if l=='otherwise' else 1.0 for l in text_data['label']] # comment out
#     labels = torch.tensor(labels, dtype=torch.long) # comment out
    Y = np.array(text_data['label'])
    labels = torch.zeros((len(Y),4)) # one-hot encodeing
    for i in range(len(Y)) :
      if (Y[i]=="individual") :
        labels[i][0] = 1
      elif (Y[i]=="community") :
        labels[i][1] = 1
      elif (Y[i]=="society") :
        labels[i][2] = 1
      else :
        labels[i][3] = 1 
    
    imgs = [Image.open(os.path.join(img_path, img_dir)).convert('RGB') for img_dir in image_dirs]
    return imgs, text_labels, labels
    
def split_dataset(imgs, labels, preprocess):
    imgs = [preprocess(d) for d in imgs]
    imgs = torch.stack(imgs)
    data = [[imgs[i],labels[i]] for i in range(len(imgs))]
    return random_split(dataset=data, lengths=[int(0.7*len(imgs)), len(imgs)-int(0.7*len(imgs))])

def get_dataset(data, text_labels):
    return ImgDataset(data, text_labels)
    
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def refine_classname(class_names):
    for i, class_name in enumerate(class_names):
        class_names[i] = class_name.lower().replace('_', ' ').replace('-', ' ')
    return class_names


def save_checkpoint(state, args, is_best=False, filename='checkpoint.pth.tar'):
    savefile = os.path.join(args.model_folder, filename)
    bestfile = os.path.join(args.model_folder, 'model_best.pth.tar')
    torch.save(state, savefile)
    if is_best:
        shutil.copyfile(savefile, bestfile)
        print ('saved best file')


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


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


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
