import uproot
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import argparse


def _read_scalar(tree, branch_name, event_number, default_value):
    if branch_name not in tree:
        return default_value
    arr = tree[branch_name].array(
        entry_start=event_number, entry_stop=event_number + 1, library="np"
    )
    if arr.size == 0:
        return default_value
    return int(arr[0])


def _read_hit_array_2d(tree, branch_name, event_number):
    if branch_name not in tree:
        raise RuntimeError(f"Missing branch: {branch_name}")

    arr = tree[branch_name].array(
        entry_start=event_number, entry_stop=event_number + 1, library="np"
    )
    if arr.shape[0] == 0:
        return None

    arr = arr[0]
    if arr.ndim == 1:
        arr = arr.reshape(1, arr.shape[0])
    return arr


def _make_shades(cmap, n, low=0.35, high=0.95):
    if n <= 0:
        return []
    if n == 1:
        return [cmap((low + high) * 0.5)]
    return [cmap(x) for x in np.linspace(low, high, n)]


def plot_event(file_name, event_number, num_detectors=62, num_elements=201):
    with uproot.open(file_name) as file:
        tree = file["tree"]

        hit_mup = _read_hit_array_2d(tree, "HitArray_mup", event_number)
        hit_mum = _read_hit_array_2d(tree, "HitArray_mum", event_number)
        if hit_mup is None or hit_mum is None:
            print(f"No data found for event {event_number}")
            return

        n_mup_tracks = _read_scalar(tree, "nMupTracks", event_number, hit_mup.shape[0])
        n_mum_tracks = _read_scalar(tree, "nMumTracks", event_number, hit_mum.shape[0])

        n_mup_tracks = max(0, min(n_mup_tracks, hit_mup.shape[0]))
        n_mum_tracks = max(0, min(n_mum_tracks, hit_mum.shape[0]))

        num_detectors = min(num_detectors, hit_mup.shape[1], hit_mum.shape[1])

        overlap_code = n_mup_tracks + n_mum_tracks + 1
        matrix = np.zeros((num_elements, num_detectors), dtype=np.int16)

        def mark_cell(y, x, code):
            current = matrix[y, x]
            if current == 0:
                matrix[y, x] = code
            else:
                matrix[y, x] = overlap_code

        for track_idx in range(n_mup_tracks):
            code = 1 + track_idx
            for det_idx in range(num_detectors):
                elem = int(hit_mup[track_idx, det_idx])
                if 1 <= elem <= num_elements:
                    y = elem - 1
                    x = det_idx
                    mark_cell(y, x, code)

        for track_idx in range(n_mum_tracks):
            code = 1 + n_mup_tracks + track_idx
            for det_idx in range(num_detectors):
                elem = int(hit_mum[track_idx, det_idx])
                if 1 <= elem <= num_elements:
                    y = elem - 1
                    x = det_idx
                    mark_cell(y, x, code)

        colors = ["white"]
        colors += _make_shades(plt.cm.Reds, n_mup_tracks)
        colors += _make_shades(plt.cm.Blues, n_mum_tracks)
        colors += ["black"]

        cmap = mcolors.ListedColormap(colors)
        max_code = overlap_code
        bounds = np.arange(-0.5, max_code + 1.5, 1.0)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        plt.figure(figsize=(10, 8))
        plt.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, origin="lower")
        plt.title(f"Event {event_number} (mup = {n_mup_tracks}, mum = {n_mum_tracks})")
        plt.xlabel(f"Detector ID (1 to {num_detectors})")
        plt.ylabel(f"Element ID (1 to {num_elements})")
        plt.xticks(np.arange(0, num_detectors + 1, 10))
        plt.yticks(np.arange(0, num_elements + 1, 20))

        legend_handles = []
        for i in range(n_mup_tracks):
            legend_handles.append(
                mpatches.Patch(color=colors[1 + i], label=f"mu+ #{i + 1}")
            )
        for i in range(n_mum_tracks):
            legend_handles.append(
                mpatches.Patch(
                    color=colors[1 + n_mup_tracks + i], label=f"mu- #{i + 1}"
                )
            )
        legend_handles.append(
            mpatches.Patch(color="black", label="overlap (any tracks)")
        )

        if len(legend_handles) <= 20:
            plt.legend(handles=legend_handles, loc="upper right")
        else:
            plt.legend(
                handles=[mpatches.Patch(color="black", label="overlap (any tracks)")],
                loc="upper right",
            )

        plot_name = os.path.basename(file_name).replace(".root", "")
        plot_name += f"_{event_number}.png"
        plot_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, plot_name)
        plt.savefig(plot_path)

        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot GT mu+/mu- hits per track (pixel), overlap -> black."
    )
    parser.add_argument("file", type=str, help="Path to the ROOT file.")
    parser.add_argument("-event", type=int, required=True, help="Event number to plot.")
    parser.add_argument(
        "--num_detectors",
        type=int,
        default=62,
        help="Number of detectors (default: 62).",
    )
    parser.add_argument(
        "--num_elements",
        type=int,
        default=201,
        help="Number of element IDs (default: 201).",
    )
    args = parser.parse_args()

    plot_event(
        args.file,
        args.event,
        num_detectors=args.num_detectors,
        num_elements=args.num_elements,
    )
