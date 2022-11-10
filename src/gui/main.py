from random import randint
import os
import eel


class DataManager:
    """Class to manage the data for the GUI."""

    def __init__(self):
        self.path = None

        self.has_labels = False
        self.has_ids = False

        self.dataset = None
        self.lables = None

        self.selected_inputs = None
        self.selected_outputs = None

    def set_path(self, path):
        """Set the path to the dataset."""
        self.path = path

    def do_stuff(self):
        """Do stuff with the data."""
        print(f"do stuff in {self.path}")
        return "done"


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
        eel.expose(self.do_stuff)
        eel.expose(self.data_manager_do_stuff)

    def do_stuff(self):
        """Do some stuff."""
        print("Doing stuff")
        return randint(0, 10000)

    def data_manager_do_stuff(self):
        """Do some stuff."""
        print("Doing stuff 2")
        return randint(-9000, 100)


def run_app():
    """Run the app."""
    app = GUIApp()
    app.expose_all()
    app.start()


if __name__ == "__main__":
    run_app()
