import os

def divider(title: str):
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)

def find_project_root(start_path=None, markers=("data", ".git")):
    """
    Walk upward from start_path until one of the markers is found.
    """
    if start_path is None:
        start_path = os.getcwd()

    path = os.path.abspath(start_path)

    while True:
        # Check for any marker inside this folder
        if any(os.path.exists(os.path.join(path, marker)) for marker in markers):
            return path

        # Move one directory up
        new_path = os.path.dirname(path)

        # If we cannot go higher, stop
        if new_path == path:
            raise RuntimeError("Project root not found.")

        path = new_path