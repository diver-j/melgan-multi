from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import argparse
import json
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
from env import AttrDict, build_env
from meldataset import MelDataset, mel_spectrogram
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from models import Generator, MultiScaleDiscriminator, feature_loss, generator_loss, discriminator_loss
from utils import plot_spectrogram

h = None
device = None


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model.load_state_dict(checkpoint_dict['model'])
    print("Loaded checkpoint '{}' (iteration {})" .format(
          checkpoint_path, iteration))
    return model, optimizer, iteration


def save_checkpoint(model, optimizer, learning_rate, steps, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
          steps, filepath))
    torch.save({'model': model.state_dict(),
                'iteration': steps,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def fit(a, epochs):
    if h.num_gpus > 1:
        init_distributed(a.rank, h.num_gpus, a.group_name, h.dist_config['dist_backend'], h.dist_config['dist_url'])

    generator = Generator().to(device)
    discriminator = MultiScaleDiscriminator().to(device)

    if h.num_gpus > 1:
        generator = apply_gradient_allreduce(generator)
        discriminator = apply_gradient_allreduce(discriminator)

    g_optim = torch.optim.Adam(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    d_optim = torch.optim.Adam(discriminator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    steps = 0
    if a.cp_g != "" and  a.cp_d != "":
        generator, g_optim, steps = load_checkpoint(a.cp_g, generator, g_optim)
        discriminator, d_optim, steps = load_checkpoint(a.cp_d, discriminator, d_optim)
        steps += 1

    with open(a.input_train_metafile, 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                          for x in fi.read().split('\n') if len(x) > 0]

    with open(a.input_valid_metafile, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]

    trainset = MelDataset(training_files, h.segment_size, h.n_fft, h.num_mels,
                        h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=False,
                              drop_last=True)

    if a.rank == 0:
        validset = MelDataset(validation_files, h.segment_size, h.n_fft, h.num_mels,
                            h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False)
        valid_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                  sampler=None,
                                  batch_size=1,
                                  pin_memory=False,
                                  drop_last=True)

    if a.rank == 0:
        os.makedirs(a.cps, exist_ok=True)
        print("checkpoints directory : ", a.cps)
        sw = SummaryWriter(os.path.join(a.cps, 'logs'))

    epoch_offset = max(0, int(steps / len(train_loader)))
    generator.train()
    discriminator.train()
    for epoch in range(epoch_offset, epochs):
        start = time.time()

        if a.rank == 0:
            print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            start_b = time.time()
            x, y, _ = batch
            x = torch.autograd.Variable(x.to(device))
            y = torch.autograd.Variable(y.to(device))
            y = y.unsqueeze(1)

            g_optim.zero_grad()
            y_ghat = generator(x)
            y_dhat_r, y_dhat_g, fmap_r, fmap_g = discriminator(y, y_ghat)
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen = generator_loss(y_dhat_g) + loss_fm
            if h.num_gpus > 1:
                reduced_loss_gen = reduce_tensor(loss_gen.data, h.num_gpus).item()
            else:
                reduced_loss_gen = loss_gen.item()
            loss_gen.backward()
            g_optim.step()

            d_optim.zero_grad()
            y_ghat = y_ghat.detach()
            y_dhat_r, y_dhat_g, _, _ = discriminator(y, y_ghat)
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_dhat_r, y_dhat_g)
            if h.num_gpus > 1:
                reduced_loss_disc = reduce_tensor(loss_disc.data, h.num_gpus).item()
            else:
                reduced_loss_disc = loss_disc.item()
            loss_disc.backward()
            d_optim.step()

            if a.rank == 0 and steps % a.stdout_interval == 0:
                print('Steps : {:d}, Gen Loss : {:4.3f}, Disc Loss : {:4.3f}, s/b : {:4.3f}'.
                      format(steps, reduced_loss_gen, reduced_loss_disc, time.time() - start_b))

            if a.rank == 0 and steps % a.checkpoint_interval == 0 and steps != 0:
                checkpoint_path = "{}/g_{:08d}".format(a.cps, steps)
                save_checkpoint(generator, g_optim, h.learning_rate, steps, checkpoint_path)
                checkpoint_path = "{}/d_{:08d}".format(a.cps, steps)
                save_checkpoint(discriminator, d_optim, h.learning_rate, steps, checkpoint_path)

            if a.rank == 0 and steps % a.summary_interval == 0:
                sw.add_scalar("training/gen_loss", reduced_loss_gen, steps)
                sw.add_scalar("training/disc_loss", reduced_loss_disc, steps)
                for i, (r, g) in enumerate(zip(losses_disc_r, losses_disc_g)):
                    sw.add_scalar("training/disc{:d}_loss_r".format(i+1), r, steps)
                    sw.add_scalar("training/disc{:d}_loss_g".format(i+1), g, steps)
                for i, (r, g) in enumerate(zip(y_dhat_r, y_dhat_g)):
                    sw.add_histogram("training/disc{:d}_r_output".format(i+1), r, steps)
                    sw.add_histogram("training/disc{:d}_g_output".format(i+1), g, steps)
                sw.add_histogram("training/gen_output", y_ghat, steps)
                sw.add_audio('training_gt/y', y[0], steps, h.sampling_rate)
                sw.add_audio('training_predicted/y_hat', y_ghat[0], steps, h.sampling_rate)

            if a.rank == 0 and steps % a.validation_interval == 0: # and steps != 0:
                for i, batch in enumerate(valid_loader):
                    x, y, _ = batch
                    y_ghat = generator(x.to(device))

                    sw.add_audio('validation_gt/y_{}'.format(i), y[0], steps, h.sampling_rate)
                    sw.add_audio('validation_predicted/y_hat_{}'.format(i), y_ghat[0], steps, h.sampling_rate)

                    # print(plot_spectrogram(x[i]))
                    sw.add_figure('validation_gt/y_spec_{}'.format(i), plot_spectrogram(x[0]), steps)
                    y_hat_spec = mel_spectrogram(y_ghat.detach().cpu().numpy()[0][0], h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
                              h.fmin, h.fmax, center=False)
                    sw.add_figure('validation_predicted/y_hat_spec_{}'.format(i), plot_spectrogram(y_hat_spec), steps)
                    if i == 4:
                        break

            steps += 1

        if a.rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time()-start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='data/LJSpeech-1.1/wavs')
    parser.add_argument('--input_train_metafile', default='data/LJSpeech-1.1/metadata_ljspeech.csv')
    parser.add_argument('--input_valid_metafile', default='data/LJSpeech-1.1/metadata_test_ljspeech.csv')
    parser.add_argument('--inference', default=False, action='store_true')
    parser.add_argument('--cps', default='cp_melgan')
    parser.add_argument('--cp_g', default='') # ex) cp_mgt_01/g_100.pth
    parser.add_argument('--cp_d', default='') # ex) cp_mgt_01/d_100.pth
    parser.add_argument('--config', default='hparams.json')
    parser.add_argument('--training_epochs', default=5000, type=int)
    parser.add_argument('--stdout_interval', default=1, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.cps)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
        h.num_gpus = torch.cuda.device_count()
    else:
        device = torch.device('cpu')

    fit(a, a.training_epochs)


if __name__ == '__main__':
    main()

