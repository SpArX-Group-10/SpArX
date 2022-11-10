import os
import tkinter as tk
from tkinter import filedialog
import eel
import numpy as np

from user_input import import_dataset


class DataManager:
    """Class to manage the data for the GUI."""

    def __init__(self):
        self.path = None

        self.xtrain, self.ytrain = np.empty(0), np.empty(0)
        self.xlables, self.ylabels = np.empty(0), np.empty(0)

        # the index of the selected data columns
        self.selected_xlables = []

    def load_data(self, path):
        """Load the data from a file."""
        self.path = path
        self.xtrain, self.ytrain = import_dataset(path, has_index=False)
        self.xlables = self.xtrain.columns.to_numpy()
        self.ylabels = self.ytrain.columns.to_numpy()

    def set_features(self, features):
        """Set the selected features."""
        self.selected_xlables = features


class GUIApp:
    """Class to hold the eel app."""

    def __init__(self):
        self.data_manager = DataManager()
        eel.init(os.path.join(os.path.dirname(os.path.abspath(__file__)), "web"))

    def start(self):
        """Start the app."""
        eel.start("index.html", size=(800, 600))

    def expose_all(self):
        """Expose all the functions to the GUI."""
        eel.expose(self.browse_files)
        eel.expose(self.get_xlabels)
        eel.expose(self.set_xlabels)

    def browse_files(self):
        """Browse file."""
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        file_path = filedialog.askopenfile()
        file_path = file_path.name if file_path else None
        self.data_manager.load_data(file_path)
        return file_path

    def get_xlabels(self):
        """Get the x labels."""
        return self.data_manager.xlables.tolist()

    def set_xlabels(self, labels):
        """Set the x labels."""
        self.data_manager.set_features(labels)


def run_app():
    """Run the app."""
    app = GUIApp()
    app.expose_all()
    app.start()


if __name__ == "__main__":
    run_app()
