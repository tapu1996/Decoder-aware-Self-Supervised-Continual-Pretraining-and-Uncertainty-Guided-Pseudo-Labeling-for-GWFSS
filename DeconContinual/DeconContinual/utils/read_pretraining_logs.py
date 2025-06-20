import os 
from pathlib import Path
import re
import tqdm
from torch.utils.tensorboard import SummaryWriter

class LogsReader:
    def __init__(self, file_path, log_dir):
        self.file_path = file_path
        self.summary_writer = SummaryWriter(log_dir)
        self.iteration = 0
    @staticmethod
    # Function to extract necessary values from the given line format
    def extract_values(line):
        match = re.search(r'Train: \[(\d+)/\d+\]\[(\d+)/(\d+)\].*? loss (\d+\.\d+).*? enc_loss (\d+\.\d+).*? dec_loss (\d+\.\d+).*? group_loss_enc (\d+\.\d+).*? group_loss_dec (\d+\.\d+).*? contrastive_loss_enc (\d+\.\d+).*? contrastive_loss_dec (\d+\.\d+)', line)
        if match:
            epoch = int(match.group(1))
            current_iter = int(match.group(2))
            total_iters = int(match.group(3))
            losses = {
                'loss': float(match.group(4)),
                'enc_loss': float(match.group(5)),
                'dec_loss': float(match.group(6)),
                'group_loss_enc': float(match.group(7)),
                'group_loss_dec': float(match.group(8)),
                'contrastive_loss_enc': float(match.group(9)),
                'contrastive_loss_dec': float(match.group(10)),
            }
            return epoch, current_iter, total_iters, losses
        return None, None, None, None
    
    def read_logs(self):
        with open(self.file_path, 'r') as file:
            lines = file.readlines()
            for  line in tqdm.tqdm(
                lines, desc=f'Reading logs from {os.path.basename(self.file_path)}', total=len(lines)
                ):
                epoch, current_iter, total_iters, losses = self.extract_values(line)                

                if losses is not None:
                    self.iteration = (epoch - 1) * total_iters + current_iter
                    for loss_name, loss_value in losses.items():
                        self.summary_writer.add_scalar(loss_name, loss_value, global_step=self.iteration)
                        self.summary_writer.add_scalar(loss_name+'_epoch', loss_value, global_step=epoch)

        self.summary_writer.close()


if __name__ == "__main__":
    import argparse
    import glob
    parser = argparse.ArgumentParser(description='Read logs')
    parser.add_argument('-fp', '--file-path', type=str, help='Path to the log file')
    parser.add_argument('-fop', '--folder-path', type=str, help='Path of folder containing all log files')
    parser.add_argument('-o', '--output-dir', type=str, help='Path to the output directory')
    args = parser.parse_args()
    if args.file_path is None and args.folder_path is None:
        raise ValueError('Please provide either file path or folder path')
    elif args.file_path is not None :
        dir_name = os.path.basename(args.file_path).split('.')[0]
        if args.output_dir is not None:
            log_dir = os.path.join(args.output_dir, 'logs', dir_name)
        else:
            log_dir = os.path.join(Path(__file__).parents[1], 'output', 'logs', dir_name) 
        log_reader = LogsReader(file_path=args.file_path, log_dir=log_dir)
        log_reader.read_logs()
    else:
        assert args.folder_path is not None
        log_files = glob.glob(os.path.join(args.folder_path, "log*.txt"))
        for log_file in log_files:
            dir_name = os.path.basename(log_file).split('.')[0]
            if args.output_dir is not None:
                log_dir = os.path.join(args.output_dir, 'logs', dir_name)
            else:
                log_dir = os.path.join(Path(__file__).parents[1], 'output', 'logs', dir_name) 
            log_reader = LogsReader(file_path=os.path.join(args.folder_path, log_file), log_dir=log_dir)
            log_reader.read_logs()
