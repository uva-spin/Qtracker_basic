import uproot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse

def plot_event(file_name, event_number):
    # Open the ROOT file
    with uproot.open(file_name) as file:
        # Access the TTree
        tree = file["tree"]
        
        # Extract data for detectorID and elementID
        detector_id = tree["detectorID"].array(entry_start=event_number, entry_stop=event_number + 1)
        element_id = tree["elementID"].array(entry_start=event_number, entry_stop=event_number + 1)
        
        # Flatten the arrays (they are stored as vectors in ROOT)
        if len(detector_id) == 0 or len(element_id) == 0:
            print(f"No data found for event {event_number}")
            return
        detector_id = detector_id[0].tolist()
        element_id = element_id[0].tolist()
        
        # Prepare a 2D matrix (200 x 60), shifting indices to start from 1
        matrix = np.zeros((200, 60))
        for det_id, elem_id in zip(detector_id, element_id):
            det_index = det_id - 1  # Shift detector ID to start from 1
            elem_index = elem_id - 1  # Shift element ID to start from 1
            if 0 <= det_index < 60 and 0 <= elem_index < 200:
                matrix[elem_index, det_index] = 1  # Mark seen hits
        
        # Create a custom colormap
        colors = [(0, 0, 1), (1, 1, 0.8)]  # Blue for 0, Light Yellow for 1
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=2)
        
        # Plot the matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(matrix, aspect="auto", cmap=cmap, origin="lower")
        plt.title(f"Event {event_number}")
        plt.xlabel("Detector ID (1 to 60)")
        plt.ylabel("Element ID (1 to 200)")
        plt.xticks(np.arange(0, 61, 10))
        plt.yticks(np.arange(0, 201, 20))
        
        # Show the plot
        plt.show()

if __name__ == "__main__":
    # Argument parser for command-line flags
    parser = argparse.ArgumentParser(description="Plot hits for a specific event in a ROOT file.")
    parser.add_argument("file", type=str, help="Path to the ROOT file.")
    parser.add_argument("-event", type=int, required=True, help="Event number to plot.")
    args = parser.parse_args()

    # Call the plotting function
    plot_event(args.file, args.event)
