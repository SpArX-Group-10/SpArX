from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label


class ResultsTabContent(GridLayout):
    """Results tab content."""
    def __init__(self, data_manager):
        self.data_manager = data_manager
        super(ResultsTabContent, self).__init__()
        self.add_widget(Label(text='Results Tab Content'))
