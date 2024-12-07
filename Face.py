import os


def rename_files(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    print(f"Number of files in directory: {len(files)}")
    # Get the list of files in the directory
    files = sorted(files)

    # Create a range based on the number of files
    sequence = range(1, len(files) + 1)

    # Rename each file based on the sequence
    for file, new_name in zip(files, sequence):
        file_ext = os.path.splitext(file)[1]  # Get the file extension
        new_filename = f"image.{new_name}{file_ext}"
        old_path = os.path.join(directory, file)
        new_path = os.path.join(directory, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed: {file} -> {new_filename}")


def main():
    directory = input("Enter the directory containing files to rename: ")

    if os.path.isdir(directory):
        rename_files(directory)
    else:
        print("The specified directory does not exist.")


if __name__ == "__main__":
    main()
