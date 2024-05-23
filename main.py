import os
import time

from gen_scripts.script_1 import transform
from src.audio import Audio
from src.make_grid import make_grid

if __name__ == "__main__":
    # Wait for user input to put a audio file
    # audio_file = input("Enter the path to the audio file: ")
    # audio_file = "audio/all_systems_go_xenotech.aif"
    # audio_file = "audio/all_systems_go_xenotech_clipped.aif"
    # audio_file = "audio/all_systems_go_xenotech_clipped2.aif"
    audio_file = "audio/all_systems_go_xenotech_clipped3.aif"

    # Check if the file exists
    if not os.path.exists(audio_file):
        print("File not found.")
        exit()

    # Create an Audio object
    audio = Audio(audio_file)

    start_time = time.time()
    # Process the time frames
    basename = os.path.basename(audio_file)
    img_path = f"images/{time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime())}/"
    os.makedirs(img_path)
    sync = False
    # sync = True
    audio.create_visualization(make_grid, sync=sync, img_path=img_path, transform=transform)

    processing_time_seconds = time.time() - start_time
    processing_time_seconds = round(processing_time_seconds, 2)  # Round to 2 decimal places
    print(f"Processing time frames took {processing_time_seconds} seconds.")
