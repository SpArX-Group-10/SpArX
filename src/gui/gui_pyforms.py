from pyforms.basewidget import BaseWidget
from pyforms.controls   import ControlFile
from pyforms.controls   import ControlCheckBox
from pyforms.controls   import ControlSlider
from pyforms.controls   import ControlButton

# 1) Import data set 
    # check it has the index column
    # chec if it has the column name
# 2) Select features 
    # transform string labels to numbers
# 3) Spawn the interactives 

# ---------------------------------------
# Index | Feature 1 | Feature 2 | Feature ...| Feature n | Label
#  ()   |    (/)    |    (/)    |     (/)    |    (/)    |   

class WelcomeScreen(BaseWidget):

    def __init__(self, *args, **kwargs):
        super().__init__('Sparx')

        self._selectdata = ControlButton("Data")
        self._selectmodel = ControlButton("Model")

        # for i in range(10):
        #     setattr(self, f'_data{i}', ControlCheckBox(label=f'_data {i}'))

        # #Definition of the forms fields
        # self._videofile     = ControlFile('Video')
        # self._outputfile    = ControlText('Results output file')
        # self._threshold     = ControlSlider('Threshold', default=114, minimum=0, maximum=255)
        # self._blobsize      = ControlSlider('Minimum blob size', default=110, minimum=100, maximum=2000)
        # self._runbutton     = ControlButton('Run')

        # #Define the function that will be called when a file is selected
        # self._videofile.changed_event     = self.__videoFileSelectionEvent
        # #Define the event that will be called when the run button is processed
        # self._runbutton.value       = self.__runEvent

        #Define the organization of the Form Controls
        # self._formset = [
        #     ('_videofile', '_outputfile'),
        #     '_threshold',
        #     ('_blobsize', '_runbutton'),
        # ]

        # self._formset = [f'_data{i}' for i in range(10)]


    def __videoFileSelectionEvent(self):
        """
        When the videofile is selected instanciate the video in the player
        """
        self._player.value = self._videofile.value

    def __process_frame(self, frame):
        """
        Do some processing to the frame and return the result frame
        """
        return frame

    def __runEvent(self):
        """
        After setting the best parameters run the full algorithm
        """
        pass


if __name__ == '__main__':
    from pyforms import start_app
    start_app(WelcomeScreen)