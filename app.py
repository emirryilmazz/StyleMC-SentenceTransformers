"""import re
import math
import time
import click
"""
import os
from werkzeug.utils import secure_filename
import subprocess
import random


"""import cv2
import clip
import dnnlib
import numpy as np
import torch
from torch import linalg as LA
import torch.nn.functional as F
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
import PIL.Image
from PIL import Image
import matplotlib.pyplot as plt"""
from flask import Flask, redirect, url_for, request, render_template, send_file

"""
from find_direction import block_forward
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma
import id_loss as id_l
"""
# Copy from find_direction.py file

# params
"""network_pkl: str,
seeds: Optional[List[int]],
truncation_psi: float,
noise_mode: str,
outdir: str,
class_idx: Optional[int],
projected_w: Optional[str],
projected_s: Optional[str],
text_prompt: str,
resolution: int,
batch_size: int,
identity_power: str,"""

app = Flask(__name__)
"""
def generate_images(f, prompt: str, range_of_effect: int):
    find_directions(prompt)
    return 'Success'


def find_directions(prompt: str):
    network_pkl='./ffhq.pkl'
    seeds = list(range(1,129))
    truncation_psi = 0.7
    noise_mode = 'const'
    outdir = 'out'
    class_idx = None
    projected_w = None
    projected_s = None
    text_prompt = prompt
    resolution= 256
    batch_size = 1
    identity_power = 'high'

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    os.makedirs(outdir, exist_ok=True)
    # Synthesize the result of a W projection
    if projected_w is not None:
        if seeds is not None:
            print('warn: --seeds is ignored when using --projected-w')
        print(f'Generating images from projected W "{projected_w}"')
        ws = np.load(projected_w)['w']
        ws = torch.tensor(ws, device=device)  # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        for idx, w in enumerate(ws):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/proj{idx:02d}.png')
        return

    # Labels
    label = torch.zeros([1, G.c_dim], device=device).requires_grad_()
    if G.c_dim != 0:
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    model, preprocess = clip.load("ViT-B/32", device=device)
    text = clip.tokenize([text_prompt]).to(device)
    text_features = model.encode_text(text)

    # Generate images
    for i in G.parameters():
        i.requires_grad = True

    mean = torch.as_tensor((0.48145466, 0.4578275, 0.40821073), dtype=torch.float, device=device)
    std = torch.as_tensor((0.26862954, 0.26130258, 0.27577711), dtype=torch.float, device=device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)

    transf = Compose([Resize(224, interpolation=InterpolationMode.BICUBIC, ), CenterCrop(224)])

    styles_array = []
    print("seeds:", seeds)
    t1 = time.time()
    for seed_idx, seed in enumerate(seeds):
        if seed == seeds[-1]:
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        ws = G.mapping(z, label, truncation_psi=truncation_psi)

        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, G.synthesis.num_ws, G.synthesis.w_dim])
            ws = ws.to(torch.float32)

            w_idx = 0
            for res in G.synthesis.block_resolutions:
                block = getattr(G.synthesis, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        styles = torch.zeros(1, 26, 512, device=device)
        styles_idx = 0
        temp_shapes = []
        for res, cur_ws in zip(G.synthesis.block_resolutions, block_ws):
            block = getattr(G.synthesis, f'b{res}')

            if res == 4:
                temp_shape = (block.conv1.affine.weight.shape[0], block.conv1.affine.weight.shape[0],
                              block.torgb.affine.weight.shape[0])
                styles[0, :1, :] = block.conv1.affine(cur_ws[0, :1, :])
                styles[0, 1:2, :] = block.torgb.affine(cur_ws[0, 1:2, :])
                if seed_idx == (len(seeds) - 1):
                    block.conv1.affine = torch.nn.Identity()
                    block.torgb.affine = torch.nn.Identity()
                styles_idx += 2
            else:
                temp_shape = (block.conv0.affine.weight.shape[0], block.conv1.affine.weight.shape[0],
                              block.torgb.affine.weight.shape[0])
                styles[0, styles_idx:styles_idx + 1, :temp_shape[0]] = block.conv0.affine(cur_ws[0, :1, :])
                styles[0, styles_idx + 1:styles_idx + 2, :temp_shape[1]] = block.conv1.affine(cur_ws[0, 1:2, :])
                styles[0, styles_idx + 2:styles_idx + 3, :temp_shape[2]] = block.torgb.affine(cur_ws[0, 2:3, :])
                if seed_idx == (len(seeds) - 1):
                    block.conv0.affine = torch.nn.Identity()
                    block.conv1.affine = torch.nn.Identity()
                    block.torgb.affine = torch.nn.Identity()
                styles_idx += 3
            temp_shapes.append(temp_shape)

        styles = styles.detach()
        styles_array.append(styles)

    resolution_dict = {256: 6, 512: 7, 1024: 8}
    id_coeff_dict = {"high": 2, "medium": 0.5, "low": 0.1, "none": 0}
    id_coeff = id_coeff_dict[identity_power]
    styles_direction = torch.zeros(1, 26, 512, device=device)
    styles_direction_grad_el2 = torch.zeros(1, 26, 512, device=device)
    styles_direction.requires_grad_()
    global id_l
    id_loss = id_l.IDLoss("a").to(device).eval()

    temp_photos = []
    grads = []
    for i in range(math.ceil(len(seeds) / batch_size)):
        # print(i*batch_size, "processed", time.time()-t1)

        styles = torch.vstack(styles_array[i * batch_size:(i + 1) * batch_size]).to(device)
        seed = seeds[i]

        styles_idx = 0
        x2 = img2 = None

        for k, (res, cur_ws) in enumerate(zip(G.synthesis.block_resolutions, block_ws)):
            block = getattr(G.synthesis, f'b{res}')
            if k > resolution_dict[resolution]:
                continue

            if res == 4:
                x2, img2 = block_forward(block, x2, img2, styles[:, styles_idx:styles_idx + 2, :], temp_shapes[k],
                                         noise_mode=noise_mode)
                styles_idx += 2
            else:
                x2, img2 = block_forward(block, x2, img2, styles[:, styles_idx:styles_idx + 3, :], temp_shapes[k],
                                         noise_mode=noise_mode)
                styles_idx += 3

        img2_cpu = img2.detach().cpu().numpy()
        temp_photos.append(img2_cpu)
        if i > 3:
            continue

        styles2 = styles + styles_direction

        styles_idx = 0
        x = img = None
        for k, (res, cur_ws) in enumerate(zip(G.synthesis.block_resolutions, block_ws)):
            block = getattr(G.synthesis, f'b{res}')
            if k > resolution_dict[resolution]:
                continue
            if res == 4:
                x, img = block_forward(block, x, img, styles2[:, styles_idx:styles_idx + 2, :], temp_shapes[k],
                                       noise_mode=noise_mode)
                styles_idx += 2
            else:
                x, img = block_forward(block, x, img, styles2[:, styles_idx:styles_idx + 3, :], temp_shapes[k],
                                       noise_mode=noise_mode)
                styles_idx += 3

        identity_loss, _ = id_loss(img, img2)
        identity_loss *= id_coeff
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255)
        img = (transf(img.permute(0, 3, 1, 2)) / 255).sub_(mean).div_(std)
        image_features = model.encode_image(img)
        cos_sim = -1 * F.cosine_similarity(image_features, (text_features[0]).unsqueeze(0))
        (identity_loss + cos_sim.sum()).backward(retain_graph=True)

    t1 = time.time()
    styles_direction.grad[:, list(range(26)), :] = 0
    with torch.no_grad():
        styles_direction *= 0

    for i in range(math.ceil(len(seeds) / batch_size)):
        print(i * batch_size, "processed", time.time() - t1)

        seed = seeds[i]
        styles = torch.vstack(styles_array[i * batch_size:(i + 1) * batch_size]).to(device)
        img2 = torch.tensor(temp_photos[i]).to(device)
        styles2 = styles + styles_direction

        styles_idx = 0
        x = img = None
        for k, (res, cur_ws) in enumerate(zip(G.synthesis.block_resolutions, block_ws)):
            block = getattr(G.synthesis, f'b{res}')
            if k > resolution_dict[resolution]:
                continue

            if res == 4:
                x, img = block_forward(block, x, img, styles2[:, styles_idx:styles_idx + 2, :], temp_shapes[k],
                                       noise_mode=noise_mode)
                styles_idx += 2
            else:
                x, img = block_forward(block, x, img, styles2[:, styles_idx:styles_idx + 3, :], temp_shapes[k],
                                       noise_mode=noise_mode)
                styles_idx += 3

        identity_loss, _ = id_loss(img, img2)
        identity_loss *= id_coeff
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255)
        img = (transf(img.permute(0, 3, 1, 2)) / 255).sub_(mean).div_(std)
        image_features = model.encode_image(img)
        cos_sim = -1 * F.cosine_similarity(image_features, (text_features[0]).unsqueeze(0))
        (identity_loss + cos_sim.sum()).backward(retain_graph=True)

        styles_direction.grad[:, [0, 1, 4, 7, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], :] = 0

        if i % 2 == 1:
            styles_direction.data = (styles_direction - styles_direction.grad * 5)
            grads.append(styles_direction.grad.clone())
            styles_direction.grad.data.zero_()
            if i > 3:
                styles_direction_grad_el2[grads[-2] * grads[-1] < 0] += 1

    styles_direction = styles_direction.detach()
    styles_direction[styles_direction_grad_el2 > (len(seeds) / batch_size) / 4] = 0

    output_filepath = f'{outdir}/direction_' + text_prompt.replace(" ", "_") + '.npz'
    np.savez(output_filepath, s=styles_direction.cpu().numpy())

    print("time passed:", time.time() - t1)
"""


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/images')
def get_image():
    filename = request.args.get('filename')
    print(filename)
    file_path = 'out/' + filename
    return send_file(file_path, mimetype='image/gif')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # params
    f = request.files['file']
    global extension
    prompt = request.form['prompt']
    power = request.form['range']

    if f.filename != '':
        extension = f.filename.split('.')[-1]
    else:
        extension = 'jpeg'

    # check if direction exists
    npz_file = 'direction_"' + secure_filename(prompt) + '".npz'
    basepath = os.path.dirname(__file__)
    full_file_path = os.path.join(basepath, 'out', npz_file)
    direction_exists = os.path.isfile(full_file_path)
    new_file = secure_filename(prompt) + '_' + str(power) + '.' + extension
    prompt = '--text_prompt="' + prompt + '"'
    print(new_file)
    if not direction_exists:
        # find directions
        print(subprocess.run(['python3', 'find_direction.py', prompt, '--resolution=256', '--batch_size=1',
                              '--identity_power=high', '--outdir=out', '--seeds=1-129', '--network=./ffhq.pkl']))

    # check if reference image given
    if f.filename != '':
        # create latent space with reference image
        full_file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(full_file_path)
        file_path = './uploads' + '/' + f.filename
        print('reference file: ', file_path)
        print(subprocess.run(['python3', 'encoder4editing/infer.py', '--input_image', file_path]))
        print(subprocess.run(
            ['python3', 'w_s_converter.py', '--outdir=out', '--projected-w=encoder4editing/projected_w.npz',
             '--network=./ffhq.pkl']))
    else:
        # create latent space with random image (no reference is given)
        print('no reference image is given')
        random_seed = (random.randint(1, 128))
        random_seed = '--seeds=' + str(random_seed)
        print(subprocess.run(['python3', 'generate_w.py', '--trunc=0.7', random_seed, '--network=./ffhq.pkl']))
        print(subprocess.run(
            ['python3', 'w_s_converter.py', '--outdir=out', '--projected-w=encoder4editing/projected_w.npz',
             '--network=./ffhq.pkl']))
    print(subprocess.run(['python3', 'generate_fromS.py', prompt, '--change_power=' + power, '--outdir=out',
                          '--s_input=out/input.npz', '--network=./ffhq.pkl']))

    return new_file


if __name__ == '__main__':
    app.run(debug=True)
