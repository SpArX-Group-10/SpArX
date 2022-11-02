
class GUIDataManager:    
    """Class to manage the data for the GUI."""
    def __init__(self):
        self.dataset = None
        self.lables = None
        
        self.selected_inputs = None
        self.selected_outputs = None


    def reset(self):
        self.dataset = None
        self.lables = None
        
        self.selected_inputs = None
        self.selected_outputs = None

