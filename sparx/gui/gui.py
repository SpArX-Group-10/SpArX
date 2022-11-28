from functools import partial
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelHeader

from gui_data_manager import GUIDataManager
from tabs.dataset_tab import DatasetTabContent
from tabs.model_tab import ModelTabContent
from tabs.result_tab import ResultsTabContent


def CreateTabWithContent(name, content, **kwargs):
    """Creates a tab with the given content."""
    tph = TabbedPanelHeader(text=name, **kwargs)
    tph.content = content
    return tph


class ProcessTabs(TabbedPanel):
    """This class creates the tabs for the GUI"""

    def __init__(self, data_manager):
        # there are 3 tabs in the GUI, dataset, model, and results
        super().__init__()
        self.do_default_tab = False
        self.add_widget(CreateTabWithContent("Dataset", DatasetTabContent(data_manager)))
        self.add_widget(CreateTabWithContent("Model", ModelTabContent(data_manager)))
        self.add_widget(CreateTabWithContent("Results", ResultsTabContent(data_manager)))

        Clock.schedule_once(partial(self.switch_to, self.tab_list[-1]), 0)


class MyApp(App):
    """This class creates the GUI"""

    def build(self):
        """This function builds the GUI"""
        self.title = "SpArX setup"  # pylint: disable=attribute-defined-outside-init

        data_manager = GUIDataManager()
        return ProcessTabs(data_manager)


if __name__ == "__main__":
    MyApp().run()
