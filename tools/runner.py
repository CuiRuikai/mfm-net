from email.mime import base
from scipy import rand
import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()

    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    dis_optimizer, dis_scheduler = builder.build_opti_sche(base_model.module.dis_params(), config, opt_type='dis')
    gen_optimizer, gen_scheduler = builder.build_opti_sche(base_model.module.gen_params(), config, opt_type='gen')

    if args.resume:
        builder.resume_optimizer(dis_optimizer, gen_optimizer, args, logger = logger)

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['DiscLoss', 'GenLoss', 'Recon', 'Comp', 'Matching'])

        base_model.train()  # set model to training mode
        train_dataloader.dataset.shuffle_gt()  # shuffle the gt list so that training data is unpaired
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            data_time.update(time.time() - batch_start_time)
            x = data[0].cuda()
            y = data[1].cuda()

            # update discriminator
            base_model.zero_grad()  # instead of optimizer.zero_grad(), use this reference; https://discuss.pytorch.org/t/model-zero-grad-or-optimizer-zero-grad/28426
            ret = base_model.module.dis_forward(x, y)
            loss_disc = base_model.module.get_dis_loss(ret)
            loss_disc.backward()
            dis_optimizer.step()

            # update geneator
            base_model.zero_grad()
            # gen_optimizer.zero_grad()
            ret = base_model.module.gen_forward(x, y)
            loss_gen, recon_loss, com_loss, matching_loss = \
                            base_model.module.get_gen_loss(ret, (x, y))
            loss_gen.backward()
            gen_optimizer.step()

            if args.distributed:
                loss_disc = dist_utils.reduce_tensor(loss_disc, args)
                loss_gen = dist_utils.reduce_tensor(loss_gen, args)
                recon_loss = dist_utils.reduce_tensor(recon_loss, args)
                com_loss = dist_utils.reduce_tensor(com_loss, args)
                matching_loss = dist_utils.reduce_tensor(matching_loss, args)
                losses.update([loss_disc.item(), loss_gen.item(), recon_loss.item(), com_loss.item(), matching_loss.item()])
            else:
                losses.update([loss_disc.item(), loss_gen.item(), recon_loss.item(), com_loss.item(), matching_loss.item()])

            if args.distributed:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx
            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/DiscLoss', loss_disc.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/GenLoss', loss_gen.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/Recon', recon_loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/Comp', com_loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/Matching', matching_loss.item(), n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 100 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = [%.6f, %.6f]' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.5f' % l for l in losses.val()], dis_optimizer.param_groups[0]['lr'], gen_optimizer.param_groups[0]['lr']), logger = logger)

        if isinstance(dis_scheduler, list):
            for item in dis_scheduler:
                item.step()
        else:
            dis_scheduler.step()
        if isinstance(gen_scheduler, list):
            for item in gen_scheduler:
                item.step()
        else:
            gen_scheduler.step()
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/DiscLoss', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/GenLoss', losses.avg(1), epoch)
            train_writer.add_scalar('Loss/Epoch/Recon', losses.avg(2), epoch)
            train_writer.add_scalar('Loss/Epoch/Comp', losses.avg(3), epoch)
            train_writer.add_scalar('Loss/Epoch/Matching', losses.avg(4), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)

        if epoch % args.val_freq == 0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)

            # Save ckeckpoints
            if  metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, dis_optimizer, gen_optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        builder.save_checkpoint(base_model, dis_optimizer, gen_optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)
        if (config.max_epoch - epoch) < 10:
            builder.save_checkpoint(base_model, dis_optimizer, gen_optimizer,epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)
    train_writer.close()
    val_writer.close()

def validate(base_model, test_dataloader, epoch, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    # test_losses = AverageMeter(['LossL1', 'LossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]
            partial = data[0].cuda()
            gt = data[1].cuda()

            ret = base_model(partial)

            _metrics = Metrics.get(ret, gt)

            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)

            if val_writer is not None and idx % 200 == 0:
                input_pc = partial.squeeze().detach().cpu().numpy()
                input_pc = misc.get_ptcloud_img(input_pc)
                val_writer.add_image('Model%02d/Input'% idx , input_pc, epoch, dataformats='HWC')

                predicted = ret.squeeze().cpu().numpy()
                predicted_img = misc.get_ptcloud_img(predicted)
                val_writer.add_image('Model%02d/Prediction' % idx, predicted_img, epoch, dataformats='HWC')

                gt_ptcloud = gt.squeeze().cpu().numpy()
                gt_ptcloud_img = misc.get_ptcloud_img(gt_ptcloud)
                val_writer.add_image('Model%02d/GT' % idx, gt_ptcloud_img, epoch, dataformats='HWC')

            if (idx+1) % 200 == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % m for m in _metrics]), logger=logger)

        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)

    return Metrics(config.consider_metric, test_metrics.avg())


crop_ratio = {
    'easy': 1/4,
    'median' :1/2,
    'hard': 3/4
}

def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    if args.save_pred:
        print_log('Save Predictions', logger=logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger = logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP
    if args.distributed:
        raise NotImplementedError()

    test(base_model, test_dataloader, args, config, logger=logger)

def test(base_model, test_dataloader,  args, config, logger = None):

    base_model.eval()  # set model to eval mode

    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1
    if args.save_pred:
        pred_save_path = os.path.join(args.experiment_path, 'preditions')
        if not os.path.exists(pred_save_path):
            os.makedirs(pred_save_path)

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]
            partial = data[0].cuda()
            gt = data[1].cuda()

            ret = base_model(partial)

            if args.save_pred:
                misc.point_save(ret, pred_save_path, model_id, type='ply')
            _metrics = Metrics.get(ret, gt)
            test_metrics.update(_metrics)

            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)
            ret = base_model(partial)

            if (idx+1) % 200 == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % m for m in _metrics]), logger=logger)
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[TEST] Metrics = %s' % (['%.4f' % m for m in test_metrics.avg()]), logger=logger)



    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)


    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall \t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)
    return
