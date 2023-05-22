import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio

def get2domains(df):
    df['n_domains'] = df['chain-dom-bounds-seq-conds'].apply(lambda x: len(eval(x)))
    return df[(df.n_domains>1)]




def make_gif(png_dir):
    images = []
    for file_name in sorted(os.listdir(png_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(os.path.join(save_dir, 'saved_gif2_0p15.gif'), images,  duration = 0.2)

def make_from_df(df_path, save_dir, label_dir, pae_dir):
    df = pd.read_csv(df_path)
    df = get2domains(df)
    completed_paes = os.listdir(pae_dir)
    for i, row in df.iterrows():
        id = row['chain-desc'].split('|')[-1]
        if id + '.npy' not in completed_paes:
            continue
        try:
            labels = np.load(os.path.join(label_dir, id + '.npy'))
        except:
            continue
        plt.imshow(labels)
        plt.title(id)
        plt.savefig(os.path.join(save_dir, str(i)))
        # plt.show()

def make_from_folder(label_dir, save_dir):
    for i, f in enumerate(os.listdir(label_dir)):
        path = os.path.join(label_dir, f)
        labels = np.load(path)['arr_0']
        plt.imshow(labels)
        name = f.split('.')[0]
        plt.title(name)
        plt.savefig(os.path.join(save_dir, name))


pae_dir = '../../features/old_paes/'
label_dir = '../../features/pairwise'
df_path = '../../redundant_data_and_files/ds_final_imp.csv'
save_dir  = '../../labels_gifs_new_train_facebook/'
os.makedirs(save_dir, exist_ok=True)
# make_from_df(df_path, save_dir, label_dir, pae_dir)
make_from_folder(label_dir, save_dir)
make_gif(save_dir)
