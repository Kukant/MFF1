from kivy.lang import Builder
from kivy.uix.screenmanager import Screen
import json
import os.path

Builder.load_string("""
<WinnerInputScreen>:
    FloatLayout:
        Label:
            size_hint: None, None
            pos_hint:{"center_x":0.5,"center_y":0.9}
            font_size: 100
            size: 1000,150
            text_size: self.size
            text: 'Congratulations'
            halign: 'center'
            valign: 'middle'
            
        Label:
            size_hint: None, None
            pos_hint:{"center_x":0.5,"center_y":0.7}
            size: 1000,150
            text_size: self.size
            text: "Please enter the winner's name:"
            halign: 'center'
            valign: 'middle'
            
        TextInput:
            id: text_input
            size_hint: None, None
            pos_hint:{"center_x":0.5,"center_y":0.6}
            size: 300,50

        Button:
            size_hint: None, None
            pos_hint:{"center_x":0.5,"center_y":0.3}
            size: 300,150
            text: 'Save'
            on_press: 
                root.save_winner()

<HallOfFameScreen>:
    FloatLayout:
        Label:
            size_hint: None, None
            pos_hint:{"center_x":0.5,"center_y":0.9}
            font_size: 100
            size: 1000,150
            text_size: self.size
            text: 'Hall Of Fame'
            halign: 'center'
            valign: 'middle'
            
        Label:
            id: winners
            size_hint: None, None
            pos_hint:{"center_x":0.5,"top_y":0.7}
            size: 400,1000
            font_size: 40
            text_size: self.size
            text: "a lot of text"
            halign: 'left'
            valign: 'top'

        Button:
            size_hint: None, None
            pos_hint:{"center_x":0.5,"center_y":0.1}
            size: 200,100
            text: 'Restart'
            on_press: 
                root.manager.current = 'settings'
""")


class WinnerInputScreen(Screen):
    def save_winner(self):
        ScoreSaver.save(self.ids.text_input.text)
        self.manager.current = 'hall_of_fame_screen'


class HallOfFameScreen(Screen):
    def on_enter(self, *args):
        text = ""
        for i, p in enumerate(ScoreSaver.get_names(), 1):
            text += "{}. {}\n".format(i, p)
        self.ids.winners.text = text


class ScoreSaver:
    fn = 'assets/scores.json'

    @staticmethod
    def save(name):
        if os.path.exists(ScoreSaver.fn):
            with open(ScoreSaver.fn) as fr:
                data = json.load(fr)['names']
        else:
            data = []

        with open(ScoreSaver.fn, 'w') as fw:
            data.insert(0, name)
            if len(data) > 10:
                data = data[:10]
            json.dump({
                'names': data
            }, fw)

    @staticmethod
    def get_names():
        if os.path.exists(ScoreSaver.fn):
            with open(ScoreSaver.fn) as fr:
                return json.load(fr)['names']
        else:
            return []

