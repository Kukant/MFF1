from kivy.lang import Builder
from kivy.uix.screenmanager import Screen

from game import game, kc

# the layout of the first screen for selecting the number of players
Builder.load_string("""
<GameSettingsScreen>:
    FloatLayout:
        Label:
            size_hint: None, None
            pos_hint:{"center_x":0.5,"center_y":0.6}
            size: 300,150
            text_size: self.size
            text: 'Choose players count:'
            halign: 'center'
            valign: 'middle'
        Spinner:
            id: spinner
            size_hint: None, None
            pos_hint:{"center_x":0.5,"center_y":0.5}
            size: 300,100
            text: '2'
            values: '2', '3', '4', '5'
        Button:
            size_hint: None, None
            pos_hint:{"center_x":0.5,"center_y":0.3}
            size: 300,150
            text: 'Start'
            on_press: 
                root.start_game()
                root.manager.current = 'game_screen'
        
""")


class GameSettingsScreen(Screen):
    # this method is called on clicking the 'Start' button
    def start_game(self):
        kc.bind_keyboard()
        # setup the game for given number of players
        game.game_setup(int(self.ids.spinner.text))
