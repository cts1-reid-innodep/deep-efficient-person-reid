from utils.getter import *
import argparse
import os
import gc
import copy
gc.enable()

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True


def train(config):

    if config.oneshot_learning == 'yes':
        train_transforms = build_transforms(config, is_train=True)
        dataset_reference = init_dataset(config.dataset_names + '_origin',
                            root=config.root_dir)
        train_set_reference = ImageDataset(dataset_reference.train, train_transforms)
        train_loader_reference = DataLoader(
            train_set_reference, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=train_collate_fn)

    # prepare dataset
    train_loader, val_loader, num_query, num_classes = get_dataset_and_dataloader(
        config)

    # prepare model
    model = Backbone(num_classes=num_classes, model_name=config.model_name, model_path=config.pretrain_path, pretrain_choice=config.pretrain_choice, attr_lens=config.attr_lens).to(config.device)

    if config.if_with_center == 'no':
        print('Train without center loss, the loss type is',
              config.loss_type)
        optimizer = get_lr_policy(config.lr_policy, model)
        scheduler = WarmupMultiStepLR(optimizer, config.steps, config.gamma, config.warmup_factor,
                                      config.warmup_iters, config.warmup_method)
        loss_func = make_loss(config, num_classes)

        start_epoch = 0
        
        # Add for using self trained model
        if config.pretrain_choice == 'self':
            start_epoch = eval(config.pretrain_path.split(
                '/')[-1].split('.')[0].split('_')[-1])
            print('Start epoch:', start_epoch)
            path_to_optimizer = config.pretrain_path.replace(
                'model', 'optimizer')
            print('Path to the checkpoint of optimizer:', path_to_optimizer)
            model.load_state_dict(torch.load(config.pretrain_path).state_dict())
            optimizer.load_state_dict(torch.load(path_to_optimizer).state_dict())
            scheduler = WarmupMultiStepLR(optimizer, config.steps, config.gamma, config.warmup_factor,
                                          config.warmup_iters, config.warmup_method, start_epoch)

        if config.oneshot_learning == 'yes':
            do_oneshot_train(
                config,
                model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,      # modify for using self trained model
                loss_func,
                num_query,
                start_epoch,     # add for using self trained model
                dataset_reference,
                train_loader_reference,
                num_classes
            )
        else:
            do_train(
                config,
                model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,
                loss_func,
                num_query,
                start_epoch
            )

    elif config.if_with_center == 'yes':
        print('Train with center loss, the loss type is',
              config.loss_type)
        loss_func, center_criterion = make_loss_with_center(
            config, num_classes)
        optimizer, optimizer_center = get_lr_policy_with_center(
            config.lr_policy, model, center_criterion)
        scheduler = WarmupMultiStepLR(optimizer, config.steps, config.gamma, config.warmup_factor,
                                      config.warmup_iters, config.warmup_method)
        start_epoch = 0

        # Add for using self trained model
        if config.pretrain_choice == 'self':
            start_epoch = eval(config.pretrain_path.split(
                '/')[-1].split('.')[0].split('_')[-1])
            print('Start epoch:', start_epoch)
            path_to_optimizer = config.pretrain_path.replace(
                'model', 'optimizer')
            print('Path to the checkpoint of optimizer:', path_to_optimizer)
            path_to_center_param = config.pretrain_path.replace(
                'model', 'center_param')
            print('Path to the checkpoint of center_param:', path_to_center_param)
            path_to_optimizer_center = config.pretrain_path.replace(
                'model', 'optimizer_center')
            print('Path to the checkpoint of optimizer_center:',
                  path_to_optimizer_center)
            print('Model state dict: ', model.state_dict())
            print('model pretrain: ', torch.load(config.pretrain_path)._modules)
            model.load_state_dict(torch.load(config.pretrain_path).state_dict())
            optimizer.load_state_dict(torch.load(path_to_optimizer).state_dict())
            center_criterion.load_state_dict(torch.load(path_to_center_param).state_dict())
            optimizer_center.load_state_dict(
                torch.load(path_to_optimizer_center).state_dict())
            scheduler = WarmupMultiStepLR(optimizer, config.steps, config.gamma, config.warmup_factor,
                                          config.warmup_iters, config.warmup_method, start_epoch)

        if config.oneshot_learning == 'yes':
            do_oneshot_train_with_center(
                config,
                model,
                center_criterion,
                train_loader,
                val_loader,
                optimizer,
                optimizer_center,
                scheduler,
                loss_func,
                num_query,
                start_epoch,
                dataset_reference,
                train_loader_reference,
                num_classes
            )
        else:
            do_train_with_center(
                config,
                model,
                center_criterion,
                train_loader,
                val_loader,
                optimizer,
                optimizer_center,
                scheduler,      # modify for using self trained model
                loss_func,
                num_query,
                start_epoch     # add for using self trained model
            )
    else:
        print("Unsupported value for config.MODEL.IF_WITH_CENTER {}, only support yes or no!\n".format(
            config.if_with_center))

    torch.cuda.empty_cache()
    del model, optimizer, scheduler, train_loader, val_loader, loss_func
    if config.if_with_center == 'yes':
        del optimizer_center
    gc.collect()

def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]
                   ) if "WORLD_SIZE" in os.environ else 1

    config_path = os.path.join('configs', f'{args.config_file}.yaml')
    config = Config(config_path)

    output_dir = config.output_dir
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(config_path, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(config))

    if config.device == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices

    seed_everything(seed=config.seed)
    train(config)


if __name__ == '__main__':
    main()
