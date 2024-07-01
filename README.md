# sound-viz

Python tool to generate visualizations of sound files.

## Installation

Install the following:
- ffmpeg
- redis

Then, install the required python packages:
```bash
pip install -r requirements.txt
```

## Usage

Example usage with optional arguments:

```bash
python main.py path/to/your/audiofile.wav --mode screen [--size 720] [--bpm 120] [--time-signature "4/4"] [--fps 30] [--cache off] [--clear-cache off] [--async-mode off] [--async-workers 8] [--time-position "0"] [--graph "src.graphs.bubbles.Bubbles"]
```

Replace `path/to/your/audiofile.wav` with the actual path to your audio file. Optional arguments can be included or omitted based on your requirements. The values shown in brackets are the defaults and can be changed as needed.

Command line arguments:
- `filename`: The path to the audio file. This is a required argument.
- `--mode`: The mode to run the visualization in. Options are `cprofile`, `video`, `screen`. Default is `screen`.
- `--size`: The size of the visualization window in pixels. Default is `720`.
- `--bpm`: The BPM (Beats Per Minute) of the audio file. If not specified, it will be determined automatically.
- `--time-signature`: The time signature of the audio file, e.g., `4/4`. Default is `4/4`.
- `--fps`: The frames per second of the output video. Default is `30`.
- `--cache`: Whether to use caching. Options are `on`, `off`. Default is `off`.
- `--clear-cache`: Whether to clear the cache and what to clear. Options are `audio`, `graph`, `img`, `all`, `off`. Default is `off`.
- `--async-mode`: Whether to use asynchronous mode for processing. Options are `on`, `off`. Default is `off`.
- `--async-workers`: The maximum number of workers to use in async mode. Default is `8`.
- `--time-position`: The time position to render in screen mode. Can be an integer or a slice (e.g., `0:10`). If not specified, the entire file is processed.
- `--graph`: The graph class to render. Default is `src.graphs.bubbles.Bubbles`.

## License

GNU General Public License v3.0
