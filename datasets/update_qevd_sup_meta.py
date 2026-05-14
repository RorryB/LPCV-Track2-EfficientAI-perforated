import argparse
from pathlib import Path


VIDEO_EXTENSIONS = {'.avi', '.mp4'}


def parse_task_map(task_map_path: Path) -> dict[str, int]:
    class_to_label = {}
    with task_map_path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(
                    f'Invalid task map line {line_no}: expected "<label> <class_name>", got {line!r}')

            label_text, class_name = parts
            label = int(label_text)
            if class_name in class_to_label:
                raise ValueError(f'Duplicate class name in task map: {class_name}')
            class_to_label[class_name] = label

    if not class_to_label:
        raise ValueError(f'No classes found in task map: {task_map_path}')
    return class_to_label


def build_split_meta(dataset_root: Path, split: str, class_to_label: dict[str, int]) -> list[str]:
    split_root = dataset_root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f'Split directory not found: {split_root}')

    class_dirs = sorted(p for p in split_root.iterdir() if p.is_dir())
    class_names = {p.name for p in class_dirs}
    mapped_names = set(class_to_label)

    unmapped_classes = sorted(class_names - mapped_names)
    missing_classes = sorted(mapped_names - class_names)
    if unmapped_classes:
        raise ValueError(f'{split} has classes not found in task_map.txt: {unmapped_classes}')
    if missing_classes:
        raise ValueError(f'{split} is missing classes from task_map.txt: {missing_classes}')

    lines = []
    for class_dir in class_dirs:
        label = class_to_label[class_dir.name]
        videos = sorted(
            p for p in class_dir.rglob('*')
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
        if not videos:
            raise ValueError(f'No videos found for class {class_dir.name!r} in {split_root}')

        for video_path in videos:
            rel_path = video_path.relative_to(split_root).as_posix()
            lines.append(f'{rel_path} {label}')

    return lines


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'.{path.name}.tmp')
    tmp_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    tmp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate QEVD_sup train/val annotation files from task_map.txt.')
    parser.add_argument(
        '--dataset-root',
        type=Path,
        default=Path('datasets/QEVD_sup'),
        help='Root directory containing train/ and val/ class folders.')
    parser.add_argument(
        '--meta-root',
        type=Path,
        default=Path('datasets/QEVD_sup/meta'),
        help='Directory containing task_map.txt and output train.txt/val.txt.')
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val'],
        help='Dataset splits to generate.')
    args = parser.parse_args()

    class_to_label = parse_task_map(args.meta_root / 'task_map.txt')

    for split in args.splits:
        lines = build_split_meta(args.dataset_root, split, class_to_label)
        output_path = args.meta_root / f'{split}.txt'
        write_lines(output_path, lines)
        print(f'Wrote {len(lines)} entries to {output_path}')


if __name__ == '__main__':
    main()
