import hashlib
import os


def generate_unique_filename(self, filename) -> str:
    """Generate a unique filename by appending a hash if the file already exists"""
    if os.path.exists(filename):
        output_dir = os.path.dirname(filename)
        # Use the last modified file in the directory as the seed
        hash_seed = max(
            [os.path.join(output_dir, f) for f in os.listdir(output_dir)],
            key=os.path.getmtime,
        )
        hash = hashlib.md5(hash_seed.encode())
        basename, ext = os.path.splitext(filename)
        return f"{basename}_{hash.hexdigest()[:5]}{ext}"
    return filename
