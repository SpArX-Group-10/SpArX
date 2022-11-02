from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
import os


class LoadDialog(RelativeLayout):
    def __init__(self, parent):
        super(LoadDialog, self).__init__()
        self.wparent = parent

        self.file_chooser = FileChooserListView(
            path=os.getcwd(), 
            size_hint=(1, 0.9), 
            pos_hint={'y':0.1}, 
            on_submit=self.double_click_load, 
        )

        self.add_widget(self.file_chooser)

        box_layout = StackLayout(orientation='lr-tb', size_hint=(1, 0.1), spacing=10)
        box_layout.add_widget(Button(text='Cancel', size_hint=(0.5, 1), size=(300, 50), on_press=self.cancel))
        box_layout.add_widget(Button(text='Load', size_hint=(0.5, 1), size=(300, 50), on_press=self.load))
        self.add_widget(box_layout)


    def double_click_load(self, chooser, path, events):
        self.load(path)


    def load(self, path):
        self.wparent.data_manager.dataset_path = path[0]
        self.wparent.data_label.text = path[0]
        self.wparent.dismiss_popup()


    def cancel(self, events):
        self.wparent.dismiss_popup()


class DatasetTabContent(StackLayout):
    """Dataset tab content."""
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.data_label = None
        super(DatasetTabContent, self).__init__(orientation='lr-tb', padding=(10, 20))
        self.add_widgets()


    def add_row(self, elements, size_hint=(1, 1)):
        row = GridLayout(rows=1, cols=len(elements), size_hint=size_hint)
        for element in elements:
            row.add_widget(element)
        self.add_widget(row)


    def add_browse_button(self):
        elems = [
            Label(text='dataset path ...',
                size_hint=(1, None),
                size=(200, 20)
            ),
            Button(
                text='Browse Dataset', 
                on_release=self.browse_press, 
                size_hint=(None, None), 
                size=(200, 20)
            )
        ]
        self.data_label = elems[0]
        self.add_row(elems)


    def add_widgets(self):
        # add padding
        self.add_browse_button()


    def browse_press(self, e):
        self._popup = Popup(title="Load file", content=LoadDialog(self), size_hint=(1, 1))
        self._popup.open()

    def dismiss_popup(self):
        self._popup.dismiss()


