from kivy.core.window import Window
from kivy.uix.widget import Widget


class KeyboardController(Widget):
    def __init__(self):
        super().__init__()
        self.keyboard = None
        self.bind_keyboard()
        self.keys_pressed = set()
        self.controls_active = False

    def bind_keyboard(self):
        if self.keyboard is None:
            self.keyboard = Window.request_keyboard(self.on_keyboard_closed, self)
            self.keyboard.bind(on_key_down=self.on_key_down)
            self.keyboard.bind(on_key_up=self.on_key_up)

    def activate_controls(self):
        self.controls_active = True

    def deactivate_controls(self):
        self.controls_active = False

    def on_keyboard_closed(self):
        self.keyboard.unbind(on_key_down=self.on_key_down)
        self.keyboard.unbind(on_key_up=self.on_key_up)
        self.keyboard = None

    def on_key_down(self, keyboard, keycode, text, modifiers):
        self.keys_pressed.add(keycode[1])

    def on_key_up(self, keyboard, keycode):
        text = keycode[1]
        if text in self.keys_pressed:
            self.keys_pressed.remove(text)