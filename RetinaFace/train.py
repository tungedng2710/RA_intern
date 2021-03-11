from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50, cfg_re18
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.retinaface import RetinaFace
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from ptflops import get_model_complexity_info
import cv2
import numpy as np
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms
from widerface_evaluate.evaluation import evaluation


def show_params_gflops(model, size, print_layer=True, input_constructor=None):
    ''' Calculate and show params gflops of model'''
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, size,
                                                 print_per_layer_stat=print_layer,
                                                 input_constructor=input_constructor)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def distillation_loss(out, dad_out):
    loss = 0
    for i in range(len(out)):
        loss += torch.mean(torch.abs(out[i] - dad_out[i]))
    return loss


def train():
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    dataset = WiderFaceDetection(training_dataset, preproc(img_dim, rgb_mean))
    val_dataset = WiderFaceDetection(training_dataset, preproc(img_dim, rgb_mean), phase='val')

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    for iteration in tqdm(range(start_iter, max_iter)):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(
                dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=detection_collate))
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
                ckpt_path = save_folder + \
                    cfg['name'] + '_epoch_' + str(epoch) + '.pth'
                torch.save(net.state_dict(), ckpt_path)
                # print('Validating....')
                # net.eval()
                # txt_save_folder = val_to_text()
                # aps = evaluation(
                #     txt_save_folder, 'widerface_evaluate/ground_truth/')
                # writer.add_scalar('Val AP/easy_AP', aps[0], epoch)
                # writer.add_scalar('Val AP/medium_AP', aps[1], epoch)
                # writer.add_scalar('Val AP/hard_AP', aps[2], epoch)
                val(val_dataset, epoch)
                net.train()
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(
            optimizer, gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]
        # print('targets', len(targets))
        # print('images', images.shape)
        # for i in range(len(targets)):
        #     print(targets[i].shape)

        # forward
        out = net(images)
        # dad_out = dad_net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        # loss_dis = distillation_loss(out, dad_out)
        # loss_l, loss_c, loss_landm = 0 * loss_dis, 0 * loss_dis, 0 * loss_dis
        loss = loss_l + loss_c + loss_landm
        # loss += 5 * loss_dis
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        if iteration % 100 == 0:
            writer.add_scalar('Train loss/loss_total', loss.item(), iteration)
            writer.add_scalar('Train loss/loss_bbox', loss_l.item(), iteration)
            writer.add_scalar('Train loss/loss_class',
                              loss_c.item(), iteration)
            writer.add_scalar('Train loss/loss_lm',
                              loss_landm.item(), iteration)
            # writer.add_scalar('Train loss/loss_dis',
            #                   loss_dis.item(), iteration)
            writer.add_scalar('LR', lr, iteration)
            if iteration % 1000 == 0:
                print('Epoch:{}/{} || Iter: {}/{} || Loss_bbox: {:.4f} Loss_class: {:.4f} Loss_lm: {:.4f} || Batchtime: {:.4f} s || ETA: {}'
                      .format(epoch, max_epoch,
                              iteration + 1, max_iter,
                              loss_l.item(),
                              loss_c.item(),
                              loss_landm.item(),
                              batch_time,
                              str(datetime.timedelta(seconds=eta))))

    torch.save(net.state_dict(), save_folder + cfg['name'] + '_Final.pth')


@torch.no_grad()
def val(val_dataset, epoch):
    net.eval()
    batch_iterator = iter(data.DataLoader(val_dataset, val_batch_size,
                                          shuffle=False, num_workers=num_workers, collate_fn=detection_collate))
    list_loss_l = []
    list_loss_c = []
    list_loss_landm = []
    list_loss_dis = []
    list_loss = []

    for images, targets in tqdm(batch_iterator):
        # load train data
        # images, targets = next(batch_iterator)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]

        # forward
        out = net(images)
        # dad_out = dad_net(images)
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        # loss_dis = distillation_loss(out, dad_out)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm

        list_loss_l.append(loss_l.item())
        list_loss_c.append(loss_c.item())
        list_loss_landm.append(loss_landm.item())
        # list_loss_dis.append(loss_dis.item())
        list_loss.append(loss.item())

    writer.add_scalar('Val loss/loss_total',
                      sum(list_loss)/len(list_loss), epoch)
    writer.add_scalar('Val loss/loss_bbox',
                      sum(list_loss_l)/len(list_loss_l), epoch)
    writer.add_scalar('Val loss/loss_class',
                      sum(list_loss_c)/len(list_loss_c), epoch)
    writer.add_scalar('Val loss/loss_lm',
                      sum(list_loss_landm)/len(list_loss_landm), epoch)
    # writer.add_scalar('Val loss/loss_dis',
    #                   sum(list_loss_dis)/len(list_loss_dis), epoch)
    print('Validation || Loss_bbox: {:.4f} Loss_class: {:.4f} Loss_lm: {:.4f}'
          .format(loss_l.item(), loss_c.item(), loss_landm.item()))


@torch.no_grad()
def val_to_text():
    txt_origin_size = True
    txt_confidence_threshold = 0.02
    txt_nms_threshold = 0.4
    txt_save_folder = args.save_folder + 'widerface_txt/'
    testset_list = 'data/widerface/val/wider_val.txt'
    testset_folder = 'data/widerface/val/images/'
    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()

    # testing begin
    for i, img_name in enumerate(tqdm(test_dataset)):
        image_path = testset_folder + img_name
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)

        # testing scale
        target_size = 1600
        max_size = 2150
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if txt_origin_size:
            resize = 1

        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize,
                             fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor(
            [img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.cuda()
        scale = scale.cuda()

        net.phase = 'test'
        loc, conf, landms = net(img)  # forward pass
        net.phase = 'train'

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.cuda()
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0),
                              prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.cuda()
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > txt_confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
            np.float32, copy=False)
        keep = py_cpu_nms(dets, txt_nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        # dets = dets[:args.keep_top_k, :]
        # landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)

        # --------------------------------------------------------------------
        save_name = txt_save_folder + img_name[:-4] + ".txt"
        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        with open(save_name, "w") as fd:
            bboxs = dets
            file_name = os.path.basename(save_name)[:-4] + "\n"
            bboxs_num = str(len(bboxs)) + "\n"
            fd.write(file_name)
            fd.write(bboxs_num)
            for box in bboxs:
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])
                confidence = str(box[4])
                line = str(x) + " " + str(y) + " " + str(w) + \
                    " " + str(h) + " " + confidence + " \n"
                fd.write(line)
    return txt_save_folder


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retinaface Training')
    parser.add_argument(
        '--training_dataset', default='./data/widerface/train/label.txt', help='Training dataset directory')
    parser.add_argument(
        '--val_dataset', default='./data/widerface/val/label.txt', help='Val dataset directory')
    parser.add_argument('--network', default='mobile0.25',
                        help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--lr', '--learning-rate', default=1e-3,
                        type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--resume_net', default=None,
                        help='resume net for retraining')
    parser.add_argument('--resume_epoch', default=0,
                        type=int, help='resume iter for retraining')
    parser.add_argument('--weight_decay', default=5e-4,
                        type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--save_folder', default='./weights/',
                        help='Location to save checkpoint models')

    args = parser.parse_args()
    args.save_folder = args.save_folder + '_' + str(time.time()) + '/'
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
        os.mkdir(args.save_folder + 'widerface_txt/')

    writer = SummaryWriter(args.save_folder)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    elif args.network == "resnet18":
        cfg = cfg_re18

    rgb_mean = (104, 117, 123)  # bgr order
    num_classes = 2
    img_dim = cfg['image_size']
    num_gpu = cfg['ngpu']
    batch_size = cfg['batch_size']
    val_batch_size = cfg['val_batch_size']
    max_epoch = cfg['epoch']
    gpu_train = cfg['gpu_train']

    num_workers = args.num_workers
    momentum = args.momentum
    weight_decay = args.weight_decay
    initial_lr = args.lr
    gamma = args.gamma
    training_dataset = args.training_dataset
    val_datasset = args.val_dataset
    save_folder = args.save_folder

    # dad_net = RetinaFace(cfg=cfg_re50)
    # print('Loading Daddy network...')
    # state_dict = torch.load('weights/Resnet50_Final.pth')
    # # create new OrderedDict that does not contain `module.`
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     head = k[:7]
    #     if head == 'module.':
    #         name = k[7:]  # remove `module.`
    #     else:
    #         name = k
    #     new_state_dict[name] = v
    # dad_net.load_state_dict(new_state_dict)
    # dad_net.eval()
    # for p in dad_net.parameters():
    #     p.requires_grad = False

    son_net = RetinaFace(cfg=cfg)
    print("Printing net...")
    print('Son_net:')
    show_params_gflops(son_net, (3, 840, 840), False)
    # print('Dad_net:')
    # show_params_gflops(dad_net, (3, 840, 840), False)
    net = son_net

    if args.resume_net is not None:
        print('Loading resume network...')
        state_dict = torch.load(args.resume_net)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)

    if num_gpu > 1 and gpu_train:
        net = torch.nn.DataParallel(net).cuda()
        dad_net = torch.nn.DataParallel(dad_net).cuda()
    else:
        # dad_net = dad_net.cuda()
        net = net.cuda()

    cudnn.benchmark = True

    optimizer = optim.SGD(net.parameters(), lr=initial_lr,
                          momentum=momentum, weight_decay=weight_decay)
    criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

    priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.cuda()
    train()
