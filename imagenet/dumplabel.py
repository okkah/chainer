import os
import pickle

DATA_DIR = os.path.join('..', 'data')

def main():
    dirs = os.listdir(DATA_DIR)

    target_dirs = []
    for dir in dirs:
        if os.path.isdir(os.path.join(DATA_DIR, dir)):
            target_dirs.append(dir)

    dirs = [dir for dir in target_dirs]

    target_all_files = []
    for dir in dirs:
        target_dir = os.path.join(DATA_DIR, dir)
        print(target_dir)

        files = os.listdir(target_dir)

        target_files = []
        for file in files:
            if file.endswith('.bmp'):
                if dir == 'clark_total_bmp':
                    target_files.append((os.path.join(target_dir, file), 0))
                elif dir == 'mm_total_bmp':
                    target_files.append((os.path.join(target_dir, file), 1))

        print(target_files[:10])
        print(files[:10])
        target_all_files.extend(target_files)

    with open('image-label.pkl', 'wb') as wf:
        pickle.dump(target_all_files, wf)

    with open('image-label.pkl', 'rb') as rf:
        image_label = pickle.load(rf)

if __name__ == '__main__':
    main()
