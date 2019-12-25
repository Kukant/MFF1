import math
from math import cos, sin
import numpy as np
from noise import pnoise1
from random import seed, randint

from kivy.uix.widget import Widget
from kivy.graphics import *
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.image import Image
from kivy.uix.screenmanager import Screen
from kivy.core.audio import SoundLoader

seed()


class ScorchedEarthGame(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_event_type("on_step")
        Clock.schedule_interval(self._on_step, 0)

        self.active_player = None
        self.terrain = None
        self.terrain_map = None
        self.players = None

    def game_setup(self, players_count):
        # clear the canvas
        self.canvas.clear()
        self.active_player = None

        # terrain setup
        self.terrain = RandomTerrain()
        t = self.terrain.terrain
        self.terrain_map = {t[x]: t[x + 1] for x in range(0, len(t), 2)}

        # players setup
        player_x_positions = [int(Window.width/(players_count + 1) * i) for i in range(1, players_count + 1)]
        self.players = [TankPlayer((posx, self.terrain_map[posx])) for posx in player_x_positions]

        # add terrain
        self.canvas.add(self.terrain.instruction)

        # add players
        [self.canvas.add(p.instruction) for p in self.players]

        self.next_player()

    def on_step(self, diff: float):
        pass

    def next_player(self):
        if self.active_player is None:  # start of the game
            self.active_player = 0
            self.players[0].activate()
            return

        self.players[self.active_player].deactivate()

        # check count of alive players, if more than one is alive, the game continues
        if len([x for x in self.players if x.is_alive]) <= 1:
            Clock.schedule_once(self.next_screen, 2)
        else:
            # select next alive player
            for i in range(1, len(self.players)):
                x = (i + self.active_player) % len(self.players)
                if self.players[x].is_alive:
                    self.players[x].activate()
                    self.active_player = x
                    break

    def next_screen(self, _):
        self.parent.manager.current = 'winner_input'

    def _on_step(self, diff: float):
        self.dispatch("on_step", diff)


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


class Missile:
    def __init__(self):
        self.sound = SoundLoader.load('assets/explosion.wav')
        self.gravity = (0, -0.1)
        self.drag = 0.9
        self.color = Color(0, 1, 0, 0)  # set it to be invisible for now
        self.speed = (0, 0)
        self.radius = 10
        self.size = (self.radius * 2, self.radius * 2)
        self.body = Ellipse(pos=(Window.width/5, Window.height - 600), size=self.size)
        self.instruction = InstructionGroup()
        self.instruction.add(self.color)
        self.instruction.add(self.body)
        self.active = False
        self.animation = Image(
            source='assets/explosion.zip',
            anim_delay=-1,
            anim_loop=1,
            allow_stretch=True,
            keep_ratio=False
        )
        self.trajectory = Line(points=[], width=1)
        self.instruction.add(Color(1, 1, 1, .3))
        self.instruction.add(self.trajectory)
        self.animation_color = Color(1, 1, 1, 0)
        self.instruction.add(self.animation_color)
        self.animation.bind(texture=self.update_texture)
        self.animation_rectangle = Rectangle(texture=self.animation.texture, pos=self.body.pos, size=(150, 150))
        self.instruction.add(self.animation_rectangle)
        game.bind(on_step=lambda x, y: self.on_step(x, y))

    def update_texture(self, instance, value):
        self.animation_rectangle.texture = value

    def activate(self):
        self.trajectory.points.clear()
        self.active = True
        self.color.a = 1

    def deactivate(self):
        self.active = False
        self.color.a = 0

    def on_step(self, sender, diff):
        if not self.active:
            return
        self.trajectory.points = self.trajectory.points + [self.body.pos[0] + self.radius, self.body.pos[1] + self.radius]
        pos = tuple(map(lambda x, y: (x + y), self.body.pos, self.speed))
        self.body.pos = pos
        # apply gravity on speed
        self.speed = tuple(map(lambda x, y: (x + y), self.gravity, self.speed))
        # apply drag
        self.speed = tuple(map(lambda x: x * 0.999, self.speed))

        if self.collides:
            self.deactivate()
            game.next_player()
            self.detonate()

    def detonate(self):
        self.sound.play()
        self.animation_color.a = 1
        self.animation._coreimage.anim_reset(True)
        self.animation.anim_delay = 0.08
        self.animation_rectangle.pos = tuple(map(lambda x: x - 75, self.body.pos))

    @property
    def collides(self) -> bool:
        # check collision with walls
        pos = self.body.pos
        if pos[0] < 0 or pos[0] + self.size[0] > Window.width:
            self.speed = (self.speed[0] * -1, self.speed[1])
            return False
        elif pos[1] + self.size[1] > Window.height:
            self.speed = (self.speed[0], self.speed[1] * -1)
            return False

        # check collision with terrain
        tm = game.terrain_map
        missile_centre = np.array((pos[0] + self.radius, pos[1] + self.radius))
        for x in range(int(pos[0]), int(pos[0]) + self.radius, 2):
            if np.linalg.norm(np.array((x, tm.get(x, 0))) - missile_centre) < self.radius:
                return True

        # check collision with players
        missile_start = (pos[0], pos[1] + self.radius * 2)
        missile_end = (pos[0] + self.radius * 2, pos[1])
        for p in game.players:
            p_pos = p.body.pos
            p_start = (p_pos[0], p_pos[1] + p.size[1])
            p_end = (p_pos[0] + p.size[0], p_pos[1])
            if self.rectangles_collide(missile_start, missile_end, p_start, p_end):
                p.missile_hit()
                return True
        return False

    @staticmethod
    def rectangles_collide(a1, a2, b1, b2) -> bool:
        # if one rectangle is on left side of other
        if a1[0] > b2[0] or b1[0] > a2[0]:
            return False
        # if one rectangle is above other
        if a1[1] < b2[1] or b1[1] < a2[1]:
            return False
        return True


class TankPlayer:
    def __init__(self, position: tuple):
        super().__init__()
        self.health = 100
        self.size = (50, 30)
        self.active = False

        # initialize the body
        pos = (position[0] - self.size[0]/2, position[1])
        instruction_group = InstructionGroup()
        self.green_color = (.1, 0.5, 0.1)
        self.body_color = Color(*self.green_color)
        instruction_group.add(self.body_color)
        self.body = Rectangle(pos=pos, size=self.size)
        instruction_group.add(self.body)

        # add the cannon and power triangle
        self.cannon_size = (7, 30)
        self.cannon_pos = (pos[0] + self.size[0]/2 - self.cannon_size[0]/2, position[1] + self.size[1])
        rotation_origin = (self.cannon_pos[0] + self.cannon_size[0]/2, self.cannon_pos[1])
        self.cannon_rotation = Rotate(origin=rotation_origin, angle=0)
        instruction_group.add(PushMatrix())
        instruction_group.add(self.cannon_rotation)

        # missile power triangle
        self.missile_power = 2.0
        self.power_rising = True
        lb = (self.cannon_pos[0] - 15 + self.cannon_size[0]/2, pos[1] + 80)
        triangle_points = [lb[0], lb[1], 30 + lb[0], lb[1], 15 + lb[0], 50 + lb[1]]
        self.triangle = Triangle(points=triangle_points)
        self.triangle_color = Color(1, 1, 1, 0)
        instruction_group.add(self.triangle_color)
        instruction_group.add(self.triangle)
        # cannon continues
        instruction_group.add(Color(0.5, 0.5, 0.5))
        instruction_group.add(Rectangle(pos=self.cannon_pos, size=self.cannon_size))
        instruction_group.add(PopMatrix())
        self.sound = SoundLoader.load('assets/blast.wav')

        # add missile for this player
        self.missile = Missile()
        instruction_group.add(self.missile.instruction)
        self.instruction = instruction_group
        game.bind(on_step=self.on_step)

        # health bar
        instruction_group.add(Color(0.8, 0.8, 0.8))
        hb_size = (self.size[0], 12)
        hb_pos = (pos[0], pos[1] - 20)
        instruction_group.add(Rectangle(pos=hb_pos, size=hb_size))
        self.hb_color = Color(0, 1, 0)
        instruction_group.add(self.hb_color)
        self.hb = Rectangle(pos=hb_pos, size=hb_size)
        instruction_group.add(self.hb)

    def activate(self):
        self.triangle_color.a = 0.22
        self.active = True

    def deactivate(self):
        self.triangle_color.a = 0
        self.active = False

    @property
    def is_alive(self) -> bool:
        return self.health > 0

    def missile_hit(self):
        self.health -= 20
        if not self.is_alive:
            game.canvas.remove(self.instruction)

        if 66 > self.health > 33:
            self.hb_color.rgb = (1, 1, 0)
        elif self.health <= 33:
            self.hb_color.rgb = (1, 0, 0)

        self.hb.size = (self.size[0] * self.health / 100, self.hb.size[1])

        self.body_color.rgb = (1, 0, 0)
        Clock.schedule_once(self.set_color_back, 1)

    def set_color_back(self, _):
        self.body_color.rgb = self.green_color

    def on_step(self, sender, diff):
        if not self.active:  # only active player can move etc
            return

        if "d" in kc.keys_pressed and self.cannon_rotation.angle > -90:
            self.cannon_rotation.angle -= 0.7
        elif "a" in kc.keys_pressed and self.cannon_rotation.angle < 90:
            self.cannon_rotation.angle += 0.7
        elif "w" in kc.keys_pressed:
            self.change_power(add=True)
        elif "s" in kc.keys_pressed:
            self.change_power(add=False)  # subtract
        elif "spacebar" in kc.keys_pressed:
            kc.keys_pressed.remove("spacebar")
            self.shoot()
            self.deactivate()

    def change_power(self, add=True):
        max = 20
        min = 2
        power_change = 0.2

        if add and self.missile_power < max:
            self.missile_power += power_change
            self.move_triangle(1)
        elif not add and self.missile_power > min:
            self.missile_power -= power_change
            self.move_triangle(-1)

    def move_triangle(self, diff: float):
        curr_points = self.triangle.points
        for i, p in enumerate(curr_points):
            if i % 2 != 0:
                curr_points[i] += diff
        self.triangle.points = curr_points

    def get_cannon_unit_vector(self) -> tuple:
        # convert to radians, rotate 90 degrees
        angle = (self.cannon_rotation.angle + 90) * math.pi / 180.0
        return cos(angle), sin(angle)

    def shoot(self):
        power = self.missile_power
        unit_vector = self.get_cannon_unit_vector()
        power_vector = tuple(map(lambda x: x*power, unit_vector))
        self.missile.activate()
        self.missile.speed = power_vector

        # calculate missile initial position
        # multiply the current unit vector  with the cannon size
        # then add the cannon position(top center)
        # finally subtract the ball radius
        initial_position = map(lambda x, y, z: x * self.cannon_size[1] + y - z,
                               unit_vector,
                               [self.cannon_pos[0] + self.cannon_size[0]/2, self.cannon_pos[1]],
                               (10, 10))
        self.missile.body.pos = initial_position
        self.sound.play()


class RandomTerrain:
    """ Class for generating random terrain using perlin noise. """
    def __init__(self):
        self.seed = randint(0, 1023)
        instruction_group = InstructionGroup()
        self.terrain = self.get_terrain()
        instruction_group.add(Line(points=self.terrain, width=1))
        # fill in the area under the line with rectangles
        for i, x in enumerate(self.terrain[1::2]):
            instruction_group.add(Rectangle(pos=(i, 0), size=(1, x)))
        self.instruction = instruction_group

    def get_terrain(self) -> list:
        terrain = []
        step = 1
        for x in range(0, Window.width + step, step):
            terrain.append(x)
            f = pnoise1(x/Window.width*2, octaves=6, base=self.seed, repeat=2048)
            f = f * Window.height/2 + Window.height * 0.5
            terrain.append(int(f))
        return terrain


class GameScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.game = game
        self.add_widget(self.game)


game = ScorchedEarthGame()
kc = KeyboardController()

