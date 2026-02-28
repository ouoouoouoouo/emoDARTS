""" Training augmented model — emoDARTS 論文修正版

修正重點：
1. UAR 改用 utils.uar()（per-class recall mean）
2. 訓練資料使用全部 train_data（對齊作者原始 code），shuffle=True
3. 支援 --genotype_path 從 search.py 輸出的 .pt 檔讀取 genotype
4. 結束時輸出 results.json（供 run_all_folds.py 彙整用）
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import utils
from config import AugmentConfig
from constant import DEVICE
from models.augment_cnn import AugmentCNN
from genotypes import Genotype

config = AugmentConfig()
device = torch.device(DEVICE)

writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)


# ── Search 結果 Genotypes（新版搜尋結果，各 fold 獨立）─────────────────────
_NORMAL_134 = [
    [('dil_conv_3x3', 0), ('sep_conv_5x5', 1)],
    [('sep_conv_3x3', 1), ('avg_pool_3x3', 0)],
    [('dil_conv_3x3', 1), ('dil_conv_3x3', 0)],
    [('sep_conv_3x3', 3), ('skip_connect', 1)],
]

FOLD_GENOTYPES = {
    1: Genotype(
        normal=_NORMAL_134,
        normal_concat=range(2, 6),
        reduce=[
            [('avg_pool_3x3', 1), ('sep_conv_5x5', 0)],
            [('avg_pool_3x3', 2), ('avg_pool_3x3', 1)],
            [('avg_pool_3x3', 2), ('avg_pool_3x3', 3)],
            [('avg_pool_3x3', 2), ('avg_pool_3x3', 4)],
        ],
        reduce_concat=range(2, 6),
        rnn=[
            [('rnn_1', 0), ('rnn_2', 1)],
            [('rnn_1', 0), ('rnn_1', 1)],
            [('rnn_1', 1), ('rnn_1', 0)],
            [('rnn_1', 1), ('rnn_1', 0)],
        ],
        rnn_concat=range(2, 6)
    ),
    2: Genotype(
        normal=[
            [('max_pool_3x3', 1), ('dil_conv_5x5', 0)],
            [('dil_conv_5x5', 2), ('dil_conv_5x5', 0)],
            [('dil_conv_3x3', 2), ('sep_conv_3x3', 3)],
            [('dil_conv_3x3', 4), ('dil_conv_5x5', 3)],
        ],
        normal_concat=range(2, 6),
        reduce=[
            [('dil_conv_3x3', 1), ('max_pool_3x3', 0)],
            [('avg_pool_3x3', 0), ('sep_conv_5x5', 1)],
            [('dil_conv_3x3', 3), ('dil_conv_3x3', 1)],
            [('dil_conv_5x5', 3), ('sep_conv_3x3', 0)],
        ],
        reduce_concat=range(2, 6),
        rnn=[
            [('rnn_3', 1), ('rnn_1', 0)],
            [('rnn_4', 2), ('rnn_3', 1)],
            [('skip_connect', 0), ('skip_connect', 1)],
            [('rnn_3', 4), ('rnn_att_2', 1)],
        ],
        rnn_concat=range(2, 6)
    ),
    3: Genotype(
        normal=[
            [('sep_conv_5x5', 1), ('dil_conv_5x5', 0)],
            [('dil_conv_3x3', 1), ('dil_conv_3x3', 0)],
            [('dil_conv_3x3', 1), ('sep_conv_3x3', 3)],
            [('dil_conv_3x3', 2), ('dil_conv_3x3', 4)],
        ],
        normal_concat=range(2, 6),
        reduce=[
            [('max_pool_3x3', 0), ('dil_conv_3x3', 1)],
            [('avg_pool_3x3', 0), ('avg_pool_3x3', 2)],
            [('sep_conv_3x3', 3), ('dil_conv_3x3', 1)],
            [('sep_conv_3x3', 0), ('dil_conv_5x5', 3)],
        ],
        reduce_concat=range(2, 6),
        rnn=[
            [('rnn_3', 0), ('rnn_4', 1)],
            [('rnn_1', 2), ('lstm_att_1', 0)],
            [('rnn_4', 2), ('rnn_4', 0)],
            [('rnn_att_1', 4), ('rnn_2', 2)],
        ],
        rnn_concat=range(2, 6)
    ),
    4: Genotype(
        normal=_NORMAL_134,
        normal_concat=range(2, 6),
        reduce=[
            [('avg_pool_3x3', 1), ('sep_conv_5x5', 0)],
            [('avg_pool_3x3', 1), ('avg_pool_3x3', 2)],
            [('avg_pool_3x3', 2), ('avg_pool_3x3', 3)],
            [('avg_pool_3x3', 1), ('avg_pool_3x3', 3)],
        ],
        reduce_concat=range(2, 6),
        rnn=[
            [('lstm_att_1', 1), ('rnn_3', 0)],
            [('lstm_att_2', 1), ('rnn_att_1', 2)],
            [('rnn_1', 0), ('lstm_att_1', 2)],
            [('rnn_1', 1), ('lstm_4', 3)],
        ],
        rnn_concat=range(2, 6)
    ),
    5: Genotype(
        normal=[
            [('sep_conv_5x5', 1), ('dil_conv_5x5', 0)],
            [('dil_conv_3x3', 1), ('dil_conv_5x5', 0)],
            [('sep_conv_3x3', 3), ('dil_conv_5x5', 0)],
            [('dil_conv_3x3', 2), ('avg_pool_3x3', 4)],
        ],
        normal_concat=range(2, 6),
        reduce=[
            [('max_pool_3x3', 0), ('dil_conv_3x3', 1)],
            [('max_pool_3x3', 0), ('avg_pool_3x3', 2)],
            [('dil_conv_3x3', 1), ('sep_conv_3x3', 3)],
            [('dil_conv_5x5', 3), ('sep_conv_3x3', 0)],
        ],
        reduce_concat=range(2, 6),
        rnn=[
            [('rnn_att_1', 0), ('lstm_2', 1)],
            [('lstm_att_1', 0), ('lstm_att_1', 1)],
            [('rnn_4', 2), ('rnn_att_2', 0)],
            [('lstm_att_1', 1), ('lstm_2', 2)],
        ],
        rnn_concat=range(2, 6)
    ),
}


def load_genotype(config) -> Genotype:
    """
    Genotype 載入優先順序：
      1. --genotype_path 指向 search.py 輸出的 .pt 檔
      2. FOLD_GENOTYPES[fold]
      3. config.genotype
    """
    genotype_path = getattr(config, 'genotype_path', None)

    if genotype_path and os.path.exists(genotype_path):
        ckpt = torch.load(genotype_path, map_location='cpu')
        genotype = ckpt['genotype']
        search_uar = ckpt.get('uar', None)
        logger.info("=" * 80)
        logger.info(f"[Genotype] 從 .pt 檔載入 — fold {config.fold}")
        logger.info(f"  Path      : {genotype_path}")
        if search_uar is not None:
            logger.info(f"  Search UAR: {search_uar:.2f}%")
        logger.info(f"  Genotype  : {genotype}")
        logger.info("=" * 80)
        return genotype

    if config.fold in FOLD_GENOTYPES:
        genotype = FOLD_GENOTYPES[config.fold]
        logger.info("=" * 80)
        logger.info(f"[Genotype] 使用 FOLD_GENOTYPES — fold {config.fold}")
        logger.info(f"  Genotype  : {genotype}")
        logger.info("=" * 80)
        return genotype

    genotype = config.genotype
    logger.info("=" * 80)
    logger.info(f"[Genotype] 使用 config.genotype — fold {config.fold}")
    logger.info(f"  Genotype  : {genotype}")
    logger.info("=" * 80)
    return genotype


def main():
    logger.info("Logger is set - augment start")
    logger.info(f"▶ Fold = {config.fold}")

    torch.cuda.set_device(config.gpus[0])
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = True

    # ── 1. 載入資料
    input_size, input_channels, n_classes, train_data, valid_data = utils.get_data(
        config.dataset, config.data_path, config.cutout_length,
        validation=True, features=config.features, fold=config.fold
    )

    logger.info(f"Data — train: {len(train_data)} | test: {len(valid_data)}")

    # ── 2. 載入 genotype
    genotype = load_genotype(config)

    # ── 3. 建立 final model
    dataset_label_weights = list(train_data.get_class_weights().values())
    criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(dataset_label_weights).to(device)
    ).to(device)

    use_aux = config.aux_weight > 0.
    model = AugmentCNN(
        input_size, input_channels, config.init_channels, n_classes,
        config.layers, use_aux, genotype, n_layers_rnn=config.rnn_layers
    )

    if len(config.gpus) > 1:
        model = nn.DataParallel(model, device_ids=config.gpus).to(device)
    else:
        model = model.to(device)

    mb_params = utils.param_size(model)
    logger.info("Model size = {:.3f} MB".format(mb_params))
    writer.add_text("Model Size", "{:.3f} MB".format(mb_params))
    writer.add_text("Num Parameters", "{}".format(utils.num_parameters(model)))
    writer.add_text("Genotype", str(genotype))

    optimizer = torch.optim.SGD(
        model.parameters(), config.lr,
        momentum=config.momentum, weight_decay=config.weight_decay
    )

    # ── 全部 train_data，shuffle=True（對齊作者原始 code）
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers, pin_memory=True, drop_last=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers, pin_memory=True, drop_last=True
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config.epochs
    )

    # ── 4. Training loop
    best_uar = 0.
    best_wa  = 0.

    for epoch in range(config.epochs):
        lr_scheduler.step()
        drop_prob = config.drop_path_prob * epoch / config.epochs
        model.drop_path_prob(drop_prob)

        train(train_loader, model, optimizer, criterion, epoch)

        cur_step = (epoch + 1) * len(train_loader)
        val_uar, val_wa = validate(
            valid_loader, model, criterion, epoch, cur_step,
            valid_data.get_class_weights()
        )

        is_best = val_uar > best_uar
        if is_best:
            best_uar = val_uar
        if val_wa > best_wa:
            best_wa = val_wa

        utils.save_checkpoint(model, config.path, is_best)
        print("")

    logger.info("Final best UAR = {:.2f}%".format(best_uar))
    logger.info("Final best WA  = {:.2f}%".format(best_wa))
    writer.add_text("Final Best UAR", "{:.2f}%".format(best_uar))
    writer.add_text("Final Best WA",  "{:.2f}%".format(best_wa))

    # ── 5. 輸出 results.json
    results = {
        "fold":    config.fold,
        "uar":     round(best_uar, 4),
        "wa":      round(best_wa,  4),
        "n_train": len(train_data),
        "n_test":  len(valid_data),
    }
    results_path = os.path.join(config.path, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved → {results_path}")


def train(train_loader, model, optimizer, criterion, epoch):
    losses    = utils.AverageMeter()
    uar_meter = utils.AverageMeter()

    cur_step = epoch * len(train_loader)
    cur_lr   = optimizer.param_groups[0]['lr']
    logger.info("Epoch {} LR {}".format(epoch, cur_lr))
    writer.add_scalar('train/lr', cur_lr, cur_step)

    model.train()

    for step, (X, y) in enumerate(train_loader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        N = X.size(0)

        optimizer.zero_grad()
        logits, aux_logits = model(X)
        loss = criterion(logits, y)
        if config.aux_weight > 0.:
            loss += config.aux_weight * criterion(aux_logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        batch_uar = utils.uar(logits, y)
        losses.update(loss.item(), N)
        uar_meter.update(batch_uar, N)

        if step % config.print_freq == 0 or step == len(train_loader) - 1:
            logger.info(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} "
                "Loss {losses.avg:.3f} UAR {uar.avg:.2f}%".format(
                    epoch + 1, config.epochs, step, len(train_loader) - 1,
                    losses=losses, uar=uar_meter
                )
            )

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/uar',  batch_uar,   cur_step)
        cur_step += 1

    logger.info("Train: [{:3d}/{}] Final UAR {:.2f}%".format(
        epoch + 1, config.epochs, uar_meter.avg))


def validate(valid_loader, model, criterion, epoch, cur_step, class_weights):
    losses    = utils.AverageMeter()
    uar_meter = utils.AverageMeter()
    wa_meter  = utils.AverageMeter()

    model.eval()
    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits, _ = model(X)
            loss = criterion(logits, y)

            batch_uar = utils.uar(logits, y)
            uar_meter.update(batch_uar, N)

            wa, _, _, _ = utils.scores(logits, y, class_weights)
            wa_meter.update(wa * 100.0, N)

            losses.update(loss.item(), N)

            if step % config.print_freq == 0 or step == len(valid_loader) - 1:
                logger.info(
                    "Valid: [{:3d}/{}] Step {:03d}/{:03d} "
                    "Loss {losses.avg:.3f} UAR {uar.avg:.2f}% WA {wa.avg:.2f}%".format(
                        epoch + 1, config.epochs, step, len(valid_loader) - 1,
                        losses=losses, uar=uar_meter, wa=wa_meter
                    )
                )

    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/uar',  uar_meter.avg, cur_step)
    writer.add_scalar('val/wa',   wa_meter.avg,  cur_step)

    logger.info("Valid: [{:3d}/{}] Final UAR {:.2f}%  WA {:.2f}%".format(
        epoch + 1, config.epochs, uar_meter.avg, wa_meter.avg))

    return uar_meter.avg, wa_meter.avg


if __name__ == "__main__":
    main()