import re
import matplotlib.pyplot as plt
import os
import sys

def plot_losses_from_log(folder_path):
    # Construct the log file path
    log_file_path = os.path.join(folder_path, "log.txt")
    
    # Initialize data storage
    epochs = []
    losses = []
    enc_losses = []
    dec_losses = []
    
    # Regex pattern to extract relevant information
    log_pattern = re.compile(r"\[([\d/]+)\s([\d:]+)\] slotcon INFO: Train: \[(\d+)/\d+\]\[\d+/\d+\].*?loss ([\d.]+).*?enc_loss ([\d.]+).*?dec_loss ([\d.]+)")
    
    try:
        # Read and parse the log file
        with open(log_file_path, "r") as file:
            for line in file:
                match = log_pattern.search(line)
                if match:
                    epoch = int(match.group(3))
                    loss = float(match.group(4))
                    enc_loss = float(match.group(5))
                    dec_loss = float(match.group(6))
                    
                    # Store extracted values
                    epochs.append(epoch)
                    losses.append(loss)
                    enc_losses.append(enc_loss)
                    dec_losses.append(dec_loss)
        
        # Plot the losses
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, losses, label="Total Loss")
        plt.plot(epochs, enc_losses, label="Encoder Loss")
        plt.plot(epochs, dec_losses, label="Decoder Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Epoch-wise Losses")
        plt.legend()
        plt.grid(True)
        
        # Save the plot as an image
        output_image_path = os.path.join(folder_path, os.path.basename(folder_path) + "_epoch_wise_losses.png")
        plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved as {output_image_path}")
    
    except FileNotFoundError:
        print(f"Log file not found at {log_file_path}. Please provide a valid folder path.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage: pass the folder path as a command-line argument
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <folder_path>")
    else:
        folder_path = sys.argv[1]
        plot_losses_from_log(folder_path)
