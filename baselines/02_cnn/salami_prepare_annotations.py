"""
    From Salami's original folder structure, copy the relevant annotation files.
"""
import tqdm
import os
import shutil


def main(path_salami, path_out):
    FN_ANNOTATION_1 = 'textfile1_uppercase.txt'
    FN_ANNOTATION_2 = 'textfile2_uppercase.txt'

    # create output folder
    os.makedirs(path_out, exist_ok=True)

    for cur_song in tqdm.tqdm(os.listdir(path_salami)):
        try:
            shutil.copy(os.path.join(path_salami, cur_song, 'parsed', FN_ANNOTATION_1),
                        os.path.join(path_out, '{}.txt'.format(cur_song)))
        except FileNotFoundError:
            shutil.copy(os.path.join(path_salami, cur_song, 'parsed', FN_ANNOTATION_2),
                        os.path.join(path_out, '{}.txt'.format(cur_song)))

        # broken annotation
        # Details: https://github.com/DDMAL/salami-data-public/issues/16
        if cur_song == '382':
            shutil.copy(os.path.join(path_salami, cur_song, 'parsed', FN_ANNOTATION_2),
                        os.path.join(path_out, '{}.txt'.format(cur_song)))


if __name__ == '__main__':
    main(os.path.join('data', 'salami-data-public', 'annotations'),
         os.path.join('data', 'salami_annotations'))
