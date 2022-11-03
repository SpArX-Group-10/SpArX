
class GUIDataManager:
    """Class to manage the data for the GUI."""
    def __init__(self):
        self.path = None

        self.has_labels = False
        self.has_ids = False

        self.dataset = None
        self.lables = None



        self.selected_inputs = None
        self.selected_outputs = None
