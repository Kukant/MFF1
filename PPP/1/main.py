
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition

from game import ScorchedEarthGame, KeyboardController, GameScreen
from settings import GameSettingsScreen


sm = ScreenManager(transition=FadeTransition())
sm.add_widget(GameSettingsScreen(name='settings'))
sm.add_widget(GameScreen(name='game_screen'))


class GameApp(App):
    def build(self):
        return sm


if __name__ == '__main__':
    GameApp().run()
