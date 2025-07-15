import argparse
from pathlib import Path

def parse():
    parser = argparse.ArgumentParser(description='Collects failed jobs from status files')
    parser.add_argument('--srcpath', type=str, help='Path to simulation status files (suffixed .status)', default=None, required=True)
    parser.add_argument('--tgtpath', type=str, help='Path to simulation status files (suffixed .status)', default=None, required=True)
    parser.add_argument('--minseed', type=int, help='Minimum seed value to consider',default=None)
    parser.add_argument('--maxseed', type=int, help='Maximum seed value to consider',default=None)

    args = parser.parse_args()
    return args

def main():
    args = parse()

    for statuspath in Path(args.srcpath).glob('*.status'):
        print(statuspath)
        with open(statuspath, 'r') as sf:
            for line in sf:
                if 'FINISHED' in line.split('|')[1]:
                    continue
                print(line.split('|'))


if __name__ == "__main__":
    main()
