import os

from flask import Flask, redirect, url_for, request, render_template, send_file
import os
from werkzeug.utils import secure_filename
import subprocess
import random

from remove_specials import remove_specials

app = Flask(__name__)


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

    extension = 'jpeg'

    # check if direction exists
    npz_file = 'direction_"' + remove_specials(prompt) + '".npz'
    basepath = os.path.dirname(__file__)
    full_file_path = os.path.join(basepath, 'out', npz_file)
    direction_exists = os.path.isfile(full_file_path)
    new_file = remove_specials(prompt) + '_' + str(power) + '.' + extension
    prompt = '--text_prompt="' + prompt + '"'
    print('new file', new_file)
    print('new file name', npz_file)
    if not direction_exists:
        # find directions
        print(subprocess.run(['python3', 'find_direction.py', prompt, '--resolution=256', '--batch_size=1',
                              '--identity_power=high', '--outdir=out', '--seeds=1-129', '--network=./ffhq.pkl']))

    # check if reference image given
    if f.filename != '':
        # create latent space with reference image
        print('file given')
        full_file_path = os.path.join(basepath, 'uploads', remove_specials(f.filename))
        print('future file path', full_file_path)
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