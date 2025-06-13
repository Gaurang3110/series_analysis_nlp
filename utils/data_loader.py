from glob import glob
import pandas as pd

def load_subtitles_dataset(dataset_path):
    from glob import glob
    import pandas as pd

    subtitles_paths = glob(dataset_path + '/*.ass')

    scripts = []
    episode_num = []

    for path in subtitles_paths:
        try:
            # Try reading with utf-8 encoding
            with open(path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
        except UnicodeDecodeError:
            # Fallback to latin1 if utf-8 fails
            with open(path, 'r', encoding='latin1') as file:
                lines = file.readlines()

        # Skip header lines
        lines = lines[27:]
        # Extract only the actual text content
        lines = [",".join(line.split(',')[9:]) for line in lines]
        lines = [line.replace('\\N', " ") for line in lines]
        script = " ".join(lines)

        # Extract episode number from filename
        episode = int(path.split('-')[-1].split('.')[0].strip())

        scripts.append(script)
        episode_num.append(episode)

    df = pd.DataFrame.from_dict({"episode": episode_num, "script": scripts})
    return df
