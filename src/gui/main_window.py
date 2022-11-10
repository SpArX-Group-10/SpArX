from pyforms.basewidget import BaseWidget
from pyforms.controls   import ControlLabel
from pyforms.controls   import ControlButton


class SimpleWindow(BaseWidget):
    def __init__(self, text, pcb, ncb, ccb, *args ,**kwargs):
        super().__init__(f'Simple Window {text}')
        self._previous_button = ControlButton('Previous')
        self._next_button = ControlButton('Next')
        self._label = ControlLabel(text)
        self._formset = [
            ('_label', '_next_button', '_previous_button')
        ]

        self._previous_button.value = pcb
        self._next_button.value = ncb

        self.close = ccb



class MainWindow(BaseWidget):
    def __init__(self, *args, **kwargs):
        super().__init__("Main Window")

        self.windows = [SimpleWindow(f'Window {i}', self.previous_window, self.next_window, self.close) for i in range(10)]
        self.windows = [self] + self.windows
        self.current_window_index = 0

        self._next_button = ControlButton('Next')
        self._formset = [
            ('_next_button')
        ]
        self._next_button.value = self.next_window

    def next_window(self):
        if self.current_window_index >= len(self.windows) - 1:
            return
        self.windows[self.current_window_index].hide()
        self.current_window_index += 1
        self.windows[self.current_window_index].show()

    def previous_window(self):
        if self.current_window_index <= 0:
            return
        self.windows[self.current_window_index].hide()
        self.current_window_index -= 1
        self.windows[self.current_window_index].show()



if __name__ == '__main__':
    from pyforms import start_app
    start_app(MainWindow, geometry=(100, 100, 800, 600))