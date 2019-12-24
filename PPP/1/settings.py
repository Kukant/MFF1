from kivy.lang import Builder
from kivy.uix.screenmanager import Screen

from game import game

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
            size_hint: None, None
            pos_hint:{"center_x":0.5,"center_y":0.5}
            size: 300,100
            text: '2'
            values: '2', '3', '4', '5'
            on_text: root.on_spinner_select(self.text)
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
    def on_spinner_select(self, text):
        game.players_count = int(text)

    def start_game(self):
        game.game_setup()
