import argparse
import csv
import itertools
import json
import random
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import UnpairedDataset, images, read_image, to_image
from .models import Generator, Discriminator, initialize


def device_for(name):
    if name == 'auto':
        name = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable. Use an ARM64 Python and MPS-enabled PyTorch on a supported Mac, or explicitly select --device cpu.')
    print(f'Device: {name}; PyTorch: {torch.__version__}', flush=True)
    return torch.device(name)


class ReplayPool:
    def __init__(self, capacity=50):
        self.capacity, self.items = capacity, []

    def query(self, batch):
        result = []
        for image in batch.detach():
            image = image.unsqueeze(0)
            if len(self.items) < self.capacity:
                self.items.append(image.cpu().clone())
            elif random.random() > 0.5:
                index = random.randrange(self.capacity)
                old = self.items[index].to(batch.device)
                self.items[index] = image.cpu().clone()
                image = old
            result.append(image)
        return torch.cat(result)


def grid(rows, path, labels):
    size = rows[0][0].shape[-1]
    canvas = Image.new('RGB', (size * len(labels), 24 + size * len(rows)), 'white')
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(labels):
        draw.text((column * size + 4, 5), label, fill='black')
    for row, tensors in enumerate(rows):
        for column, tensor in enumerate(tensors):
            canvas.paste(to_image(tensor), (column * size, 24 + row * size))
    canvas.save(path)


@torch.inference_mode()
def preview(ab, ba, root, size, device, output):
    ab.eval(); ba.eval()
    rows = []
    for split, forward, backward in [('testA', ab, ba), ('testB', ba, ab)]:
        x = read_image(images(Path(root)/split)[0], size).unsqueeze(0).to(device)
        y = forward(x)
        rows.append([x[0], y[0], backward(y)[0]])
    grid(rows, output, ['Original', 'Translated', 'Reconstructed'])
    ab.train(); ba.train()


def train(args):
    device = device_for(args.device)
    requested_epochs, requested_decay = args.epochs, args.decay_start
    random.seed(args.seed); torch.manual_seed(args.seed)
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True) if args.resume else None
    if checkpoint:
        # Resume the saved architecture, dataset selection and learning-rate schedule.
        saved = checkpoint['config']
        for key in ('size', 'channels', 'blocks', 'limit', 'seed', 'epochs', 'decay_start', 'lr', 'cycle', 'identity', 'steps'):
            setattr(args, key, saved[key])
        random.seed(args.seed); torch.manual_seed(args.seed)
    if requested_epochs is not None:
        args.epochs = requested_epochs
    elif not checkpoint:
        args.epochs = 10
    if requested_decay is not None:
        args.decay_start = requested_decay
    elif requested_epochs is not None or not checkpoint:
        args.decay_start = args.epochs // 2
    if args.size < 32 or args.size % 4:
        raise ValueError('--size must be >=32 and divisible by 4')
    if not 0 <= args.decay_start < args.epochs:
        raise ValueError('--decay-start must be >=0 and less than --epochs')
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    if (run/'latest.pt').exists() and not args.resume:
        raise ValueError('Run already contains a checkpoint. Use --resume or a different --run.')
    dataset = UnpairedDataset(args.data, args.size, args.limit, args.seed)
    # Validate held-out images before spending time training.
    images(Path(args.data)/'testA'); images(Path(args.data)/'testB')
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    ab, ba = Generator(args.channels, args.blocks).to(device), Generator(args.channels, args.blocks).to(device)
    da, db = Discriminator(args.channels).to(device), Discriminator(args.channels).to(device)
    models = {'ab': ab, 'ba': ba, 'da': da, 'db': db}
    for model in models.values():
        model.apply(initialize)
    opt_g = torch.optim.Adam(itertools.chain(ab.parameters(), ba.parameters()), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(itertools.chain(da.parameters(), db.parameters()), lr=args.lr, betas=(0.5, 0.999))
    start = 0
    pool_a, pool_b = ReplayPool(), ReplayPool()
    if checkpoint:
        for key, model in models.items():
            model.load_state_dict(checkpoint[key])
        opt_g.load_state_dict(checkpoint['opt_g']); opt_d.load_state_dict(checkpoint['opt_d'])
        start = checkpoint['epoch']
        random.setstate(checkpoint['python_rng'])
        torch.set_rng_state(checkpoint['torch_rng'])
        pool_a.items, pool_b.items = checkpoint['pool_a'], checkpoint['pool_b']
    config = {key: value for key, value in vars(args).items() if key != 'func'}
    if start >= args.epochs:
        print(f'Checkpoint has {start} completed epochs; requested total is {args.epochs}. No further training needed.', flush=True)
        return
    (run/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    print(f'Train A={len(dataset.a)}, B={len(dataset.b)}; {args.size}px; epochs {start+1}..{args.epochs}', flush=True)
    l1 = nn.L1Loss()
    for epoch in range(start, args.epochs):
        began = time.monotonic()
        factor = 1 - max(0, epoch-args.decay_start) / (args.epochs-args.decay_start)
        for opt in (opt_g, opt_d):
            for group in opt.param_groups:
                group['lr'] = args.lr * factor
        totals = dict(generator=0., discriminator=0., cycle=0., identity=0.)
        steps = min(len(loader), args.steps) if args.steps else len(loader)
        progress = tqdm(itertools.islice(loader, steps), total=steps, desc=f'Epoch {epoch+1}/{args.epochs}')
        for a, b in progress:
            a, b = a.to(device), b.to(device)
            for discriminator in (da, db):
                discriminator.requires_grad_(False)
            opt_g.zero_grad(set_to_none=True)
            fake_b, fake_a = ab(a), ba(b)
            cycle = l1(ba(fake_b), a) + l1(ab(fake_a), b)
            identity = l1(ba(a), a) + l1(ab(b), b)
            loss_g = ((db(fake_b)-1)**2).mean() + ((da(fake_a)-1)**2).mean() + args.cycle*cycle + args.identity*identity
            if not torch.isfinite(loss_g):
                raise RuntimeError('Non-finite generator loss; stop and inspect configuration.')
            loss_g.backward(); opt_g.step()
            for discriminator in (da, db):
                discriminator.requires_grad_(True)
            opt_d.zero_grad(set_to_none=True)
            loss_d = 0.5 * (((da(a)-1)**2).mean() + da(pool_a.query(fake_a)).square().mean()
                            + ((db(b)-1)**2).mean() + db(pool_b.query(fake_b)).square().mean())
            if not torch.isfinite(loss_d):
                raise RuntimeError('Non-finite discriminator loss.')
            loss_d.backward(); opt_d.step()
            values = [loss_g.item(), loss_d.item(), cycle.item(), identity.item()]
            for key, value in zip(totals, values):
                totals[key] += value / steps
            progress.set_postfix(G=f'{values[0]:.3f}', D=f'{values[1]:.3f}')
        preview(ab, ba, args.data, args.size, device, run/f'epoch_{epoch+1:03d}.png')
        state = {key: model.state_dict() for key, model in models.items()}
        state.update(config=config, epoch=epoch+1, opt_g=opt_g.state_dict(), opt_d=opt_d.state_dict(),
                     python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
                     pool_a=pool_a.items, pool_b=pool_b.items)
        torch.save(state, run/'latest.tmp')
        (run/'latest.tmp').replace(run/'latest.pt')
        metrics = dict(epoch=epoch+1, seconds=round(time.monotonic()-began, 2), lr=args.lr*factor, **totals)
        log = run/'losses.csv'
        header = not log.exists()
        with log.open('a', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics))
            if header:
                writer.writeheader()
            writer.writerow(metrics)
        print(f'Saved {run}/latest.pt; epoch took {metrics["seconds"]}s', flush=True)


@torch.inference_mode()
def infer(args):
    device = device_for(args.device)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = state['config']
    size = args.size or config['size']
    if size < 32 or size % 4:
        raise ValueError('--size must be >=32 and divisible by 4')
    forward = Generator(config['channels'], config['blocks']).to(device)
    backward = Generator(config['channels'], config['blocks']).to(device)
    forward.load_state_dict(state[args.direction])
    backward.load_state_dict(state['ba' if args.direction == 'ab' else 'ab'])
    forward.eval(); backward.eval()
    files = images(args.input)
    if args.limit:
        files = files[:args.limit]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    errors, rows = [], []
    for index, path in enumerate(tqdm(files, desc='Translating')):
        x = read_image(path, size).unsqueeze(0).to(device)
        y = forward(x)
        reconstruction = backward(y)
        # Convert [-1,1] tensor error to [0,1] pixel error.
        errors.append((reconstruction-x).abs().mean().item()/2)
        to_image(y[0]).save(output/f'{index:04d}_{path.stem}_translated.png')
        if len(rows) < 8:
            rows.append([x[0].cpu(), y[0].cpu(), reconstruction[0].cpu()])
    grid(rows, output/'comparison.png', ['Original', 'Translated', 'Reconstructed'])
    report = {'images': len(files), 'direction': args.direction, 'checkpoint_epoch': state['epoch'],
              'cycle_mae_0_to_1': sum(errors)/len(errors),
              'note': 'Cycle error measures reconstruction consistency, not translation realism or accuracy.'}
    (output/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Compact CycleGAN: apple A <-> orange B')
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('train')
    training.add_argument('--data', default='data/apple2orange')
    training.add_argument('--run', default='runs/apple2orange')
    training.add_argument('--resume')
    for name, default in [('size',128), ('channels',32), ('blocks',6), ('limit',200), ('epochs',None), ('decay-start',None), ('steps',0), ('seed',42)]:
        training.add_argument('--'+name, type=int, default=default)
    for name, default in [('lr',0.0002), ('cycle',10.), ('identity',5.)]:
        training.add_argument('--'+name, type=float, default=default)
    training.set_defaults(func=train)
    inference = commands.add_parser('infer')
    inference.add_argument('--checkpoint', default='runs/apple2orange/latest.pt')
    inference.add_argument('--input', default='data/apple2orange/testA')
    inference.add_argument('--output', default='outputs/apple2orange')
    inference.add_argument('--direction', choices=['ab','ba'], default='ab')
    inference.add_argument('--size', type=int, default=0)
    inference.add_argument('--limit', type=int, default=20)
    inference.set_defaults(func=infer)
    for subparser in (training, inference):
        subparser.add_argument('--device', choices=['auto','mps','cuda','cpu'], default='mps')
    args = parser.parse_args()
    if args.command == 'train':
        if min(args.channels, args.blocks) < 1 or (args.epochs is not None and args.epochs < 1) or min(args.limit, args.steps) < 0:
            parser.error('channels, blocks, epochs must be positive; limit and steps must be nonnegative')
    elif args.limit < 0:
        parser.error('--limit must be nonnegative')
    args.func(args)


if __name__ == '__main__':
    main()
