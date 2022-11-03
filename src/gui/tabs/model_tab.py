from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label


class ModelTabContent(GridLayout):
    """Model tab content."""
    def __init__(self, data_manager):
        self.data_manager = data_manager
        super().__init__()
        self.add_widget(Label(text='Model Tab Content'))
