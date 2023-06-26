import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = "1"
import os.path as osp
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import Subset, RandomSampler, SequentialSampler, DataLoader
from transformers import LongformerConfig, LongformerTokenizerFast
from transformers import AdamW, get_linear_schedule_with_warmup
from model import LongformerCausalModel

from processor_esl import ESL_processor

from utils import set_seed, compute_f1
from sklearn.model_selection import KFold
import numpy as np
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'longformer': (LongformerConfig, LongformerCausalModel, LongformerTokenizerFast)
}


def train_epoch(args, model, train_loader, optimizer, scheduler):
    train_loss = []
    predicted_all = []
    gold_all = []
    model.train()
    for step, batch in enumerate(train_loader):
        inputs = {'enc_input_ids': batch[0].to(args.device),
                  'enc_mask_ids': batch[1].to(args.device),
                  'global_attention_mask': batch[2].to(args.device),
                  't1_pos': batch[3],
                  't2_pos': batch[4],
                  'mh_self_attention_mask': batch[5].to(args.device),
                  'target': batch[6]
                  }

        loss, opt = model(**inputs)
        loss = loss / args.gradient_accumulation_steps
        loss.backward()
        if step % args.gradient_accumulation_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        train_loss.append(loss.item())
        predicted = torch.argmax(opt, -1)
        predicted = list(predicted.cpu().numpy())
        predicted_all += predicted
        gold = [t for bt in inputs['target'] for t in bt]
        gold_all += gold
        p, r, f1 = compute_f1(gold_all, predicted_all, logger)
        if step % args.logging_steps == 0:
            logging.info('Step {}: Train Loss {} R {} P {} F {}'.format(step, np.mean(train_loss), r, p, f1))
    p, r, f1 = compute_f1(gold_all, predicted_all, logger)
    return np.mean(train_loss), r, p, f1


def test_epoch(args, model, test_loader):
    test_loss = []
    predicted_all = []
    gold_all = []
    preds = []
    golds = []
    for batch in test_loader:
        inputs = {'enc_input_ids': batch[0].to(args.device),
                  'enc_mask_ids': batch[1].to(args.device),
                  'global_attention_mask': batch[2].to(args.device),
                  't1_pos': batch[3],
                  't2_pos': batch[4],
                  'mh_self_attention_mask': batch[5].to(args.device),
                  'target': batch[6]
                  }

        loss, opt = model(**inputs)

        test_loss.append(loss.item())
        predicted = torch.argmax(opt, -1)
        predicted = list(predicted.cpu().numpy())
        predicted_all += predicted
        preds.append(predicted)
        gold = [t for bt in inputs['target'] for t in bt]
        gold_all += gold
        golds.append(gold)
    p, r, f1 = compute_f1(gold_all, predicted_all, logger)

    return preds, golds, np.mean(test_loss), r, p, f1


def valid_epoch(args, model, valid_loader):
    valid_loss = []
    predicted_all = []
    gold_all = []
    for batch in valid_loader:
        inputs = {'enc_input_ids': batch[0].to(args.device),
                  'enc_mask_ids': batch[1].to(args.device),
                  'global_attention_mask': batch[2].to(args.device),
                  't1_pos': batch[3],
                  't2_pos': batch[4],
                  'mh_self_attention_mask': batch[5].to(args.device),
                  'target': batch[6]
                  }
        loss, opt = model(**inputs)

        valid_loss.append(loss.item())
        predicted = torch.argmax(opt, -1)
        predicted = list(predicted.cpu().numpy())
        predicted_all += predicted
        gold = [t for bt in inputs['target'] for t in bt]
        gold_all += gold
    p, r, f1 = compute_f1(gold_all, predicted_all, logger)
    return np.mean(valid_loss), r, p, f1


def cross_validation(args, model_class, tokenizer, processor):
    splits = KFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)
    logger.info("train dataloader generation")
    _, train_test_features, train_test_dataset = processor.generate_dataloader('train')
    logger.info("dev dataloader generation")
    _, dev_features, dev_dataset = processor.generate_dataloader('dev')
    valid_loader = DataLoader(dev_dataset, sampler=SequentialSampler(dev_dataset), batch_size=args.batch_size,
                              collate_fn=dev_dataset.collate_fn)

    avg_test_f1, avg_test_r, avg_test_p = [], [], []
    for fold, (train_idx, test_idx) in enumerate(splits.split(train_test_dataset)):
        logger.info("Fold {}".format(fold + 1))
        train_dataset = Subset(train_test_dataset, train_idx)
        test_dataset = Subset(train_test_dataset, test_idx)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=RandomSampler(train_dataset),
                                  collate_fn=train_test_dataset.collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=1, sampler=SequentialSampler(test_dataset),
                                 collate_fn=train_test_dataset.collate_fn)

        model = model_class(args).to(args.device)
        model.pretrained_model.resize_token_embeddings(len(tokenizer))
        model.to(args.device)
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        total_steps = len(train_loader) * args.epoch
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps * args.warm_up_ratio,
                                                    num_training_steps=total_steps)
        best_valid_f1 = 0
        best_valid_test_preds, best_valid_test_golds, best_valid_test_f1, best_valid_test_r, best_valid_test_p = 0, 0, 0, 0, 0
        for epoch in range(1, args.epoch + 1):
            model.zero_grad()
            train_loss, train_r, train_p, train_f1 = train_epoch(args, model, train_loader, optimizer, scheduler)
            logging.info('Epoch{}: Train Loss {} R {} P {} F {}'.format(epoch, train_loss, train_r, train_p, train_f1))
            model.eval()
            with torch.no_grad():
                if epoch % args.test_epoch == 0:
                    test_preds, test_golds, test_loss, test_r, test_p, test_f1 = test_epoch(args, model, test_loader)
                    logging.info('Epoch{}: Test Loss {} R {} P {} F {}'.format(epoch, test_loss, test_r, test_p, test_f1))
                if epoch % args.valid_epoch == 0:
                    valid_loss, valid_r, valid_p, valid_f1 = valid_epoch(args, model, valid_loader)
                    logging.info(
                        'Epoch{}: Valid Loss {} R {} P {} F {}'.format(epoch, valid_loss, valid_r, valid_p, valid_f1))
                    if valid_f1 >= best_valid_f1:
                        best_valid_f1 = valid_f1
                        best_valid_test_f1 = test_f1
                        best_valid_test_r = test_r
                        best_valid_test_p = test_p
        logging.info('Fold {}: R {} P {} F {}'.format(fold+1, best_valid_test_r, best_valid_test_p, best_valid_test_f1))
        avg_test_f1.append(best_valid_test_f1)
        avg_test_r.append(best_valid_test_r)
        avg_test_p.append(best_valid_test_p)
    logger.info("-------- 5-Fold Avg: R {} P {} F {} ---------".format(np.mean(avg_test_r), np.mean(avg_test_p), np.mean(avg_test_f1)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", default='EventStoryLine', type=str, help='dataset name')
    parser.add_argument("--cache_path", default='./data/cached_eci', type=str)
    parser.add_argument("--model_type", default='longformer', type=str)
    parser.add_argument("--output_dir", default='./outputs_res', type=str)
    parser.add_argument("--model_name_or_path", default="./ckpts/longformer-base-4096", type=str)
    parser.add_argument("--loss_type", default='focal', type=str)
    parser.add_argument("--inter_or_intra", default='intra_and_inter', type=str)
    parser.add_argument("--k_fold", default=5, type=int)
    parser.add_argument("--y_class", default=2, type=int)
    parser.add_argument("--epoch", default=20, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--warm_up_ratio", default=0.1, type=float)
    parser.add_argument("--class_weight", default=0.75, type=float)
    parser.add_argument("--gamma", default=2, type=int)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument("--dropout_att", default=0.1, type=float)
    parser.add_argument("--dropout_emb", default=0.3, type=float)
    parser.add_argument('--logging_steps', default=20, type=int)
    parser.add_argument('--valid_epoch', default=1, type=int)
    parser.add_argument('--test_epoch', default=1, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--use_cache', default=True, action="store_true")
    args = parser.parse_args()
    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    print(f"Output full path {osp.join(os.getcwd(), args.output_dir)}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.cache_path):
        os.makedirs(args.cache_path)

    logging.basicConfig(
        filename=os.path.join(args.output_dir, 'log.txt'), \
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', \
        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO
    )
    set_seed(args.seed)

    config_class, eci_model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, add_special_tokens=True)
    tokenizer.add_tokens(['<t>', '</t>'])

    processor = ESL_processor(args, tokenizer, logger)

    logger.info("Training/evaluation parameters %s", args)
    cross_validation(args, eci_model_class, tokenizer, processor)


if __name__ == "__main__":
    main()
