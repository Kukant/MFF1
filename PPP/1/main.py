
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, FadeTransition

from game import GameScreen
from score_board import HallOfFameScreen, WinnerInputScreen
from settings import GameSettingsScreen

sm = ScreenManager(transition=FadeTransition())

class GameApp(App):
    def build(self):
        return sm


if __name__ == '__main__':
    sm.add_widget(GameSettingsScreen(name='settings'))
    sm.add_widget(GameScreen(name='game_screen'))
    sm.add_widget(WinnerInputScreen(name='winner_input'))
    sm.add_widget(HallOfFameScreen(name='hall_of_fame_screen'))
    GameApp().run()
