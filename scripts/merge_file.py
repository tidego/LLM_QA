from pathlib import Path
from config.Config import Base_DIR, RAW_DIR


def merge_md_files_to_txt(folder_path: Path, output_file: Path):
    # 获取所有.md文件，按文件名排序
    md_files = sorted([f for f in folder_path.iterdir() if f.suffix == '.md'])

    with output_file.open('w', encoding='utf-8') as outfile:
        for file in md_files:
            outfile.write(f'# {file.name}\n')
            outfile.write(file.read_text(encoding='utf-8'))
            outfile.write('\n\n')

    print(f'输出文件为：{output_file}')


if __name__ == '__main__':
    output_txt = Base_DIR / 'kg_data' / 'merge' / 'merged_output.txt'
    merge_md_files_to_txt(RAW_DIR, output_txt)
