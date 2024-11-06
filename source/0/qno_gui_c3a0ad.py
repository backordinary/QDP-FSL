# https://github.com/mtav/QNO_GUI/blob/d8b6ba6ad22360efadea76af19a15c50f04556b8/QNO_GUI.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pygame_cards.card_sprite

try:
    import sys
    import os
    import threading
    import time
    import collections
    import json
    from random import seed, randint
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
    import pygame
    import random

    from pygame_cards import game_app, controller, deck, card_holder, enums, animation, card, game_object
    from pygame_cards.card_sprite import AbstractPygameCardSprite
    import opponent

    import QNO
    from random import shuffle
    from qiskit import QuantumCircuit, execute, Aer, IBMQ
    from qiskit.quantum_info import Statevector

except ImportError as err:
    print("Fail loading a module in file:", __file__, "\n", err)
    sys.exit(2)


# Constants
STATUS_NOTIFICATION = "STATUS_NOTIFICATION"
DIALOG_TITLE = "DIALOG_TITLE"
PLAYER_PROMPT = "PLAYER_PROMPT"
CURRENT_STATE = "CURRENT_STATE"


OpponentRecord = collections.namedtuple('OpponentRecord', 'hand, info')
DealState = collections.namedtuple('DealState', 'cards_per_player, round, next_player')
SuitInfo = collections.namedtuple('SuitInfo', 'name, symbol, description, button_txt, qiskit_index')
ToggleOption = collections.namedtuple('ToggleOption', 'json_object, id_')

##### Note: Cool unicode trick to get suit symbols!
# suit_info = [
#     SuitInfo('hearts', '\N{Black Heart Suit}'),
#     SuitInfo('diamonds', '\N{Black Diamond Suit}'),
#     SuitInfo('clubs', '\N{Black Club Suit}'),
#     SuitInfo('spades', '\N{Black Spade Suit}')
# ]
suit_info = [
    SuitInfo('q1', 'q1', 'Left qbit q1 (index 1)', '1', 1),
    SuitInfo('q0', 'q0', 'Right qbit q0 (index 0)', '0', 0)
]

def load_opponents_json():
    opponents_json_path = os.path.join(os.getcwd(), 'opponents.json')
    with open(opponents_json_path, 'r') as json_file:
        json_dict = json.load(json_file)
    return json_dict

def get_img_full_path(path):
    return pygame_cards.card_sprite.get_img_full_path(path)

class PlayDirectionImage(card_holder.CardsHolder):
    playDirection = 0

    def setupImage(self, img_path):
        img = pygame.image.load(img_path)
        self.img_scaled = pygame.transform.scale(img, (163, 161))

    def toggleDirection(self):
        print('REVERSING!!!')
        self.playDirection = (self.playDirection+1)%2
    def can_drop_card(self, card_):
        """ Check if a card can be dropped into a foundation. Conditions:
        - If a pocket is empty, only an ace can be dropped
        - If a pocket is not empty, only a card of the same suit and lower by 1 rank can be dropped
        :param card_: Card object to be dropped
        :return: Boolean, True if a card can be dropped, False otherwise
        """
        return False
    def render(self, screen):
        # draw_empty_card_pocket(self, screen, 163, 161)
        img_scaled = self.img_scaled
        if self.playDirection==1:
            img_scaled = pygame.transform.flip(self.img_scaled, False, True)
        screen.blit(img_scaled, (self.pos[0], self.pos[1]))
def draw_empty_card_pocket(holder, screen, width=None, height=None):
    """ Renders empty card pocket at the position of CardHolder object
    :param holder: CardsHolder object
    :param screen: Screen object to render onto
    """
    if width is None:
        width = card_holder.CardsHolder.card_json["size"][0]
    if height is None:
        height = card_holder.CardsHolder.card_json["size"][1]
    if len(holder.cards) == 0:
        rect = (holder.pos[0], holder.pos[1],
                width,
                height)
        pygame.draw.rect(screen, (77, 77, 77), rect, 2)

class QNO_CardSprite(AbstractPygameCardSprite):
    """ Concrete pygame Card sprite class. Represents both front and back card's sprites.

    Attributes:
        card_json - The 'card' node of the settings.json. Data can be accessed via [] operator,
                    for example: CardsHolder.card_json["size"][0]
    """

    card_json = None

    def __init__(self, state, value, pos, back_up=False):
        if QNO_CardSprite.card_json is None:
            raise ValueError('QNO_CardSprite.card_json is not initialized')
        AbstractPygameCardSprite.__init__(self, pos)

        temp_image = pygame.image.load(get_img_full_path(
            self.get_image_path(state, value))).convert_alpha()
        self.image = pygame.transform.scale(temp_image, QNO_CardSprite.card_json["size"])
        self.rect = self.image.get_rect()
        self.rect[0] = pos[0]
        self.rect[1] = pos[1]

        back_img_path = QNO_CardSprite.card_json["back_sprite_file"]
        temp_image = pygame.image.load(get_img_full_path(back_img_path)).convert_alpha()
        self.back_image = pygame.transform.scale(temp_image, QNO_CardSprite.card_json["size"])
        self.back_up = back_up

    def get_render_tuple(self):
        if self.back_up:
            return self.back_image, (self.rect[0], self.rect[1])
        else:
            return self.image, (self.rect[0], self.rect[1])

    @property
    def back_up(self):
        return self._back_up

    @back_up.setter
    def back_up(self, value):
        self._back_up = value

    @staticmethod
    def get_image_path(state, value):
        path = QNO_CardSprite.card_json["front_sprite_path"]
        path += f'card_{value}_{state}.png'
        return path


class QNO_Card(game_object.GameObject):
    """ This class represents a card. """

    def __init__(self, state, value, pos, back_up=False):
        game_object.GameObject.__init__(self)
        self.state = state
        self.value = value
        self.sprite = QNO_CardSprite(state, value, pos, back_up)
        self._back_up = back_up

    @property
    def back_up(self):
        return self._back_up

    @back_up.setter
    def back_up(self, value):
        self._back_up = value
        self.sprite.back_up = self._back_up

    def get_sprite(self):
        """ Returns card's spite object
        :return: card's sprite object
        """
        return self.sprite

    # def get_back_sprite(self):
    #     return self.back_sprite

    def render(self, screen):
        """ Renders the card's sprite on a screen passed in argument
        :param screen: screen to render the card's sprite on
        """
        self.sprite.render(screen)

    def flip(self):
        """ Flips the card from face-up to face-down and vice versa """
        self.back_up = not self.back_up

    def is_clicked(self, pos):
        """ Checks if mouse click is on card
        :param pos: tuple with coordinates of mouse click (x, y)
        :return: True if card is clicked, False otherwise
        """
        return self.sprite.is_clicked(pos)

    def check_collide(self, card_=None, pos=None):
        """ Checks if current card's sprite collides with other card's sprite, or with
        an rectangular area with size of a card. Parameters card and pos are mutually exclusive.
        :param card_: Card object to check collision with
        :param pos: tuple with coordinates (x,y) - top left corner of area to check collision with
        :return: True if cards/card and area collide, False otherwise
        """
        if card_ is not None:
            return self.sprite.check_card_collide(card_.sprite)
        elif pos is not None:
            return self.sprite.check_area_collide(pos)

    def get_pos(self):
        return self.sprite.pos

    def set_pos(self, pos):
        """ Sets position of the card's sprite
        :param pos: tuple with coordinates (x, y) where the top left corner of the card
                    should be placed.
        """
        self.sprite.pos = pos
        #self.back_sprite.set_pos(pos)

    def offset_pos(self, pos):
        """ Move the card's position by the specified offset
        :param pos: tuple with coordinates (x, y) of the offset to move card
        """
        self.sprite.offset_pos(pos)
        #self.back_sprite.offset_pos(pos)

class QNO_Deck(card_holder.CardsHolder):
    """ Deck of cards. Two types of deck available: short (6..ace) and full (2..ace)"""

    def __init__(self, type_, pos, offset, last_card_callback=None):
        """
        :param type_: int value that corresponds to enum from enums.DeckType class
        :param pos: tuple with coordinates (x, y) for bottom card in the desk
        :param last_card_callback: function that should be called when the last card is
            removed from the deck
        """
        card_holder.CardsHolder.__init__(self, pos, offset, False, last_card_callback)
        self.type = type_

        start = enums.Rank.two  # full deck type by default
        if type_ == enums.DeckType.short:
            start = enums.Rank.six

        card_pos = pos
        qno_deck = QNO.buildDeck()
        for card in qno_deck:
            state = card[0]
            value = card[1]
        # for rank in range(start, enums.Rank.ace + 1):
        #     for suit in range(enums.Suit.hearts, enums.Suit.spades + 1):
            self.cards.append(QNO_Card(state, value, card_pos, True))
            card_pos = card_pos[0] + self.offset[0], card_pos[1] + self.offset[1]

    def shuffle(self):
        """ Shuffles cards in the deck randomly """
        shuffle(self.cards)
        self.update_position()

class Crazy8sController(controller.Controller):
    def __init__(self, objects_list=None, gui_interface=None, settings_json=None):
        super(Crazy8sController, self).__init__(objects_list, gui_interface, settings_json)

        self.opponents_json = load_opponents_json()

        # Set up the stockpile.
        pos = self.settings_json["stockpile"]["position"]
        offset = self.settings_json["stockpile"]["offset"]


        # card_holder.CardsHolder.card_json = self.settings_json["card"]
        # QNO_CardSprite.card_json = self.settings_json["card"]

        self.stockpile = deck.Deck(enums.DeckType.full, pos, offset, None)
        # self.stockpile = QNO_Deck(enums.DeckType.full, pos, offset, None)
        self.add_rendered_object(self.stockpile)

        # Set up the discard pile.
        pos = self.settings_json["discard"]["position"]
        offset = self.settings_json["discard"]["offset"]
        self.discard = card_holder.CardsHolder(pos, offset)
        self.add_rendered_object(self.discard)

        # Set up the player's hand.
        pos = self.settings_json["player_hand"]["position"]
        offset = self.settings_json["player_hand"]["offset"]
        self.player_hand = card_holder.CardsHolder(pos, offset, enums.GrabPolicy.can_single_grab_any)
        self.add_rendered_object(self.player_hand)

        # set up direction arrow
        pos = self.settings_json["gui"]["play_direction"]["position"]
        offset = self.settings_json["gui"]["play_direction"]["offset"]
        self.play_direction_image = PlayDirectionImage(pos, offset, enums.GrabPolicy.no_grab)
        self.play_direction_image.setupImage(self.settings_json["gui"]["play_direction"]["img_path"])
        self.add_rendered_object(self.play_direction_image)

        # Other global behavior
        self.opponent_min_delay_ms = self.settings_json["opponent_behavior"]["min_delay_ms"]
        self.opponent_max_delay_ms = self.settings_json["opponent_behavior"]["max_delay_ms"]
        self.draw_speed = self.settings_json["card"]["draw_speed"] # Speed in pixels / second
        self.move_speed = self.settings_json["card"]["move_speed"] # Speed in pixels / second

        # Set up game state.
        self.num_opponents = 0
        self.opponents = [] # Each item in the list will be an OpponentRecord tuple.
        self.dealer = 0 # index of player who is dealer; 0 is human player
        self.turn = -1 # No one's turn yet
        self.chosen_suit = None
        self.must_choose_suit = False
        self.action_lock = False
        self.bg_pulse_animation = None

        # UI
        self.gui_interface.show_button(self.settings_json["gui"]["restart_button"],
                                       self.restart_game, "Restart")
        # self.gui_interface.show_button(self.settings_json["gui"]["test_button"],
        #                                self.testSomething, "testSomething")

        # load music and sounds
        pygame.mixer.music.load(self.settings_json["sound"]["background_music"])
        self.win_sound = pygame.mixer.Sound(self.settings_json["sound"]["win_sound"])
        self.lose_sound = pygame.mixer.Sound(self.settings_json["sound"]["lose_sound"])

        # add music button
        # self.music_paused = False
        # self.gui_interface.show_button(self.settings_json["gui"]["music_pause_button"],
        #                                self.togglePauseMusic, "Pause music") # TODO: Change aspect/text of button depending on state...
        #
        # if self.settings_json["gui"]["music_at_start"]==0:
        #     self.togglePauseMusic()

        # self.gui_interface.show_button(self.settings_json["gui"]["decrease_speed_ai"],
        #                                self.decreaseSpeed_AI, "Decrease speed AI")
        # self.gui_interface.show_button(self.settings_json["gui"]["increase_speed_ai"],
        #                                self.increaseSpeed_AI, "Increase speed AI")
        # self.gui_interface.show_button(self.settings_json["gui"]["decrease_speed_animation"],
        #                                self.decreaseSpeed_animation, "Decrease speed animation")
        # self.gui_interface.show_button(self.settings_json["gui"]["increase_speed_animation"],
        #                                self.increaseSpeed_animation, "Increase speed animation")

        self.createSpeedButtons("speed_animation", [self.minSpeed_animation, self.decreaseSpeed_animation, self.increaseSpeed_animation, self.maxSpeed_animation])
        self.createSpeedButtons("speed_ai", [self.minSpeed_AI, self.decreaseSpeed_AI, self.increaseSpeed_AI, self.maxSpeed_AI])

        self.option_replenish_stack, self.option_replenish_stack_col = self.createToggleOption("option_replenish_stack", self.toggleOptionReplenishStack)
        self.option_free_drawing, self.option_free_drawing_col = self.createToggleOption("option_free_drawing", self.toggleOptionFreeDrawing)
        self.option_music_paused, self.option_music_paused_col = self.createToggleOption("option_music_paused", self.toggleOptionMusicPaused)


        self.music_started = False
        if self.settings_json["gui"]["option_music_paused"]["initial_value"]==1:
            self.pauseMusic()
        else:
            # start music
            self.startMusic()

        ##### TODO: OPTIONS:
        # end game after deck runs out
        # rank players
        # break ties with hadamard?
        # display number of cards in each stack
        # min/max speeds
        # don't allow drawing cards if you can play


    def startMusic(self):
        if not self.music_started:
            pygame.mixer.music.play(-1)
            self.music_started = True

    def createSpeedButtons(self, base, func_list):
        j = self.settings_json["gui"][base]
        [x, y, width, height] = j["position"]
        # print([x, y, width, height])
        self.gui_interface.show_label(position=[x, y, width, height], text=j["label"], text_size=j["text_size"],
                                      timeout=0, color=j["text_color"], id_=base)
        x += j["offset_from_label"]
        label_list = ['min',' - ',' + ','max']
        step = 40
        for i in range(len(label_list)):
            pos = [x+i*step, y, width, height]
            self.gui_interface.show_button(pos, func_list[i], label_list[i])

        return

    def toggleOptionReplenishStack(self):
        self.option_replenish_stack = not self.option_replenish_stack
        if self.option_replenish_stack:
            label_text = "On"
        else:
            label_text = "Off"
        j = self.option_replenish_stack_col.json_object
        position = j["label"]["position"]
        text_size = j["label"]["text_size"]
        text_color = j["label"]["text_color"]
        self.gui_interface.hide_by_id(self.option_replenish_stack_col.id_)
        self.gui_interface.show_label(position=position, text=label_text, text_size=text_size,
                                      timeout=0, color=text_color, id_=self.option_replenish_stack_col.id_)
        print(j["button_text"], ':', self.option_replenish_stack)

    def toggleOptionFreeDrawing(self):
        self.option_free_drawing = not self.option_free_drawing
        if self.option_free_drawing:
            label_text = "On"
        else:
            label_text = "Off"
        j = self.option_free_drawing_col.json_object
        position = j["label"]["position"]
        text_size = j["label"]["text_size"]
        text_color = j["label"]["text_color"]
        self.gui_interface.hide_by_id(self.option_free_drawing_col.id_)
        self.gui_interface.show_label(position=position, text=label_text, text_size=text_size,
                                      timeout=0, color=text_color, id_=self.option_free_drawing_col.id_)
        print(j["button_text"], ':', self.option_free_drawing)

    def toggleOptionMusicPaused(self):
        self.option_music_paused = not self.option_music_paused
        if self.option_music_paused:
            label_text = "On"
            self.pauseMusic()
        else:
            label_text = "Off"
            self.unpauseMusic()
        j = self.option_music_paused_col.json_object
        position = j["label"]["position"]
        text_size = j["label"]["text_size"]
        text_color = j["label"]["text_color"]
        self.gui_interface.hide_by_id(self.option_music_paused_col.id_)
        self.gui_interface.show_label(position=position, text=label_text, text_size=text_size,
                                      timeout=0, color=text_color, id_=self.option_music_paused_col.id_)
        print(j["button_text"], ':', self.option_music_paused)

    # def togglePauseMusic(self):
    #     if self.option_music_paused:
    #         self.unpauseMusic()
    #     else:
    #         self.pauseMusic()
    def pauseMusic(self):
        pygame.mixer.music.pause()
        self.option_music_paused = True

    def unpauseMusic(self):
        self.startMusic() # start if not yet started
        pygame.mixer.music.unpause()
        self.option_music_paused = False

    def createToggleOption(self, base, func):

        j = self.settings_json["gui"][f"{base}"]
        button_text = j["button_text"]
        self.gui_interface.show_button(j["button"], func, button_text)
        position = j["label"]["position"]
        text_size = j["label"]["text_size"]
        text_color = j["label"]["text_color"]
        if j["initial_value"]==0:
            label_text = "Off"
            initial_value = False
        else:
            label_text = "On"
            initial_value = True
        id_ = f"{base}_label"
        self.gui_interface.show_label(position=position, text=label_text, text_size=text_size,
                                      timeout=0, color=text_color, id_=id_)

        # return value
        return (initial_value, ToggleOption(j, id_))

    def getSpeedInfos(self):
        s=f'self.opponent_min_delay_ms: {self.opponent_min_delay_ms}'
        s+=f', self.opponent_max_delay_ms: {self.opponent_max_delay_ms}'
        s+=f', self.draw_speed: {self.draw_speed}'
        s+=f', self.move_speed: {self.move_speed}'
        return s

    def minSpeed_AI(self):
        j = self.settings_json["gui"]["speed_ai"]
        self.opponent_min_delay_ms = j["max"]
        self.opponent_max_delay_ms = j["max"]
        print('New speed:', self.getSpeedInfos())
    def maxSpeed_AI(self):
        j = self.settings_json["gui"]["speed_ai"]
        self.opponent_min_delay_ms = j["min"]
        self.opponent_max_delay_ms = j["min"]
        print('New speed:', self.getSpeedInfos())

    def increaseSpeed_AI(self):
        self.opponent_min_delay_ms -= 100
        self.opponent_min_delay_ms = max(self.opponent_min_delay_ms, 0)
        self.opponent_max_delay_ms -= 100
        self.opponent_max_delay_ms = max(self.opponent_max_delay_ms, 0)
        print('New speed:', self.getSpeedInfos())

    def decreaseSpeed_AI(self):
        self.opponent_min_delay_ms += 100
        self.opponent_min_delay_ms = max(self.opponent_min_delay_ms, 0)
        self.opponent_max_delay_ms += 100
        self.opponent_max_delay_ms = max(self.opponent_max_delay_ms, 0)
        print('New speed:', self.getSpeedInfos())

    def minSpeed_animation(self):
        j = self.settings_json["gui"]["speed_animation"]
        self.draw_speed = j["min"]
        self.move_speed = j["min"]
        print('New speed:', self.getSpeedInfos())
    def maxSpeed_animation(self):
        j = self.settings_json["gui"]["speed_animation"]
        self.draw_speed = j["max"]
        self.move_speed = j["max"]
        print('New speed:', self.getSpeedInfos())

    def increaseSpeed_animation(self):
        j = self.settings_json["gui"]["speed_animation"]
        self.draw_speed += 100
        self.draw_speed = max(self.draw_speed, j["min"])
        self.move_speed += 100
        self.move_speed = max(self.move_speed, j["min"])
        print('New speed:', self.getSpeedInfos())

    def decreaseSpeed_animation(self):
        j = self.settings_json["gui"]["speed_animation"]
        self.draw_speed -= 100
        self.draw_speed = max(self.draw_speed, j["min"])
        self.move_speed -= 100
        self.move_speed = max(self.move_speed, j["min"])
        print('New speed:', self.getSpeedInfos())

    def playWinningSound(self):
        pygame.mixer.Sound.play(self.win_sound)
    def playLosingSound(self):
        pygame.mixer.Sound.play(self.lose_sound)

    def testSomething(self):
        print('TESTING')
        # # test reverse direction
        # print('dir:', self.getPlayDirection())
        # # self.play_direction_image.playDirection = (self.play_direction_image.playDirection+1)%2
        # self.toggleDirection()
        # print('dir:', self.getPlayDirection())

        # # test game over
        # player_idx = randint(0, self.num_opponents)
        # self.game_over(player_idx)

        # test possible states
        self.test_getPossibleStates()

        # # test drawing cards
        # player_idx = randint(0, self.num_opponents)
        #
        # self.drawCardsForPlayer(15, player_idx)
        # # self.drawCardsForPlayer(3, 1)

        # test sounds
        # self.playWinningSound()
        # self.playLosingSound()

    def getPlayDirection(self):
        return self.play_direction_image.playDirection
    def setPlayDirection(self, value):
        self.play_direction_image.playDirection = value
        return self.play_direction_image.playDirection
    def toggleDirection(self):
        self.play_direction_image.toggleDirection()
        self.show_current_state()

    def QNOsetup(self):
        raise
        discards=[]
        players=[]
        playerTurn=0
        playDirection=1
        playing=True

        discards.append(qno.unoDeck.pop(0))
        currentState=discards[0][0]
        cardVal=discards[0][1]

        while currentState=="Gate":

            qno.unoDeck=buildDeck()
            random.shuffle(qno.unoDeck)
            discards.append(qno.unoDeck.pop(0))
            currentState=discards[0][0]
            cardVal=discards[0][1]

        qc=QuantumCircuit(2)

        if currentState=="|11>":
            qc.x(0)
            qc.x(1)
        elif currentState=="|10>":
            qc.x(1)
        elif currentState=="|01>":
            qc.x(0)

    def restart_game(self):
        for animation_ in self.animations:
            animation_.is_completed = True
        self.discard.move_all_cards(self.stockpile)
        self.player_hand.move_all_cards(self.stockpile)

        self.clear_opponents()
        self.dealer = 0
        self.turn = -1 # No one's turn yet
        self.chosen_suit = None
        self.must_choose_suit = False
        self.action_lock = False
        self.clear_status_notification()
        self.clear_player_prompt()
        if self.bg_pulse_animation is not None:
            self.bg_pulse_animation.is_completed = True
        self.start_game()

    def start_game(self):
        print('=====>', 'NEW GAME!')

        self.qc = QuantumCircuit(2)

        self.stockpile.shuffle()
        self.show_status_notification("Starting game...")
        self.show_num_opponents_dialog() # This will also trigger the deal() function and play the first card.

        self.setPlayDirection(0)
        self.show_current_state()

        self.debug_enabled = True

    def on_choose_num_opponents(self, num_opponents):
        self.num_opponents = num_opponents
        self.hide_num_opponents_dialog()
        self.load_opponents()
        self.dealer = randint(0, self.num_opponents)
        self.deal()

    def load_opponents(self):
        available_opponents = list(self.opponents_json)

        # Determine position of opponents' hands.
        y = self.settings_json["opponent_hand"]["position_y"]
        x_range = self.settings_json["opponent_hand"]["x_range"]
        width = x_range[1] - x_range[0]
        width_per_opponent = width / self.num_opponents
        card_width = self.settings_json["card"]["size"][0]
        offset = self.settings_json["opponent_hand"]["offset"]

        for i in range(0, self.num_opponents):
            x = x_range[0] + ((i+0.5)*width_per_opponent) - (card_width/2)
            hand = card_holder.CardsHolder((x, y), offset, enums.GrabPolicy.can_single_grab_any)
            opponent_idx = randint(0, len(available_opponents)-1)
            opponent_json = available_opponents[opponent_idx]
            opponent_info = opponent.Opponent(opponent_json)
            available_opponents.remove(opponent_json)
            self.opponents.append(OpponentRecord(hand, opponent_info))
            self.show_opponent_name(opponent_info.name, i, x)
            print('opponent_info',i , opponent_info.name)
            hand.opponent_idx = i
            hand.player_idx = i+1
            hand.opponent_name = opponent_info.name
            self.add_rendered_object(hand)

    def show_opponent_name(self, name, idx, pos_x):
        name_json = self.settings_json["opponent_hand"]["name"]
        pos = (pos_x, name_json["position_y"])
        size = name_json["text_size"]
        color = name_json["text_color"]
        label_id = f"OPPONENT_{idx}_NAME"
        self.gui_interface.hide_by_id(label_id)
        self.gui_interface.show_label(position=pos, text=name, text_size=size,
                                      timeout=0, color=color, id_=label_id)

    def show_num_opponents_dialog(self):
        """Show dialog offering the user a choice of how many opponents they
        want.
        """
        self.show_dialog_title("How many opponents would you like?")
        pos = self.settings_json["gui"]["dialog"]["content"]["position"]
        button_size = self.settings_json["gui"]["dialog"]["content"]["button_size"]
        x_base = pos[0]
        y = pos[1]
        button_margin = self.settings_json["gui"]["dialog"]["content"]["button_margin"]

        def choose(num_opponents):
            def on_choose():
                self.on_choose_num_opponents(num_opponents)
            return on_choose
        button_x = x_base + (button_size[0] * (1-1)) + (button_margin * (1-1))
        button_pos = (button_x, y)
        button_rect = [*button_pos, *button_size]
        self.gui_interface.show_button(button_rect, choose(1), f"   {1}   ",
                                        id_=f"opponent_button{1}")
        button_x = x_base + (button_size[0] * (2-1)) + (button_margin * (2-1))
        button_pos = (button_x, y)
        button_rect = [*button_pos, *button_size]
        self.gui_interface.show_button(button_rect, choose(2), f"   {2}   ",
                                        id_=f"opponent_button{2}")
        button_x = x_base + (button_size[0] * (3-1)) + (button_margin * (3-1))
        button_pos = (button_x, y)
        button_rect = [*button_pos, *button_size]
        self.gui_interface.show_button(button_rect, choose(3), f"   {3}   ",
                                        id_=f"opponent_button{3}")
        button_x = x_base + (button_size[0] * (4-1)) + (button_margin * (4-1))
        button_pos = (button_x, y)
        button_rect = [*button_pos, *button_size]
        self.gui_interface.show_button(button_rect, choose(4), f"   {4}   ",
                                        id_=f"opponent_button{4}")
        button_x = x_base + (button_size[0] * (5-1)) + (button_margin * (5-1))
        button_pos = (button_x, y)
        button_rect = [*button_pos, *button_size]
        self.gui_interface.show_button(button_rect, choose(5), f"   {5}   ",
                                        id_=f"opponent_button{5}")
        button_x = x_base + (button_size[0] * (6-1)) + (button_margin * (6-1))
        button_pos = (button_x, y)
        button_rect = [*button_pos, *button_size]
        self.gui_interface.show_button(button_rect, choose(6), f"   {6}   ",
                                        id_=f"opponent_button{6}")

    def hide_num_opponents_dialog(self):
        self.clear_dialog_title()
        self.gui_interface.hide_by_id(f"opponent_button{1}")
        self.gui_interface.hide_by_id(f"opponent_button{2}")
        self.gui_interface.hide_by_id(f"opponent_button{3}")
        self.gui_interface.hide_by_id(f"opponent_button{4}")
        self.gui_interface.hide_by_id(f"opponent_button{5}")
        self.gui_interface.hide_by_id(f"opponent_button{6}")

    def getPlayerIndexFromHand(self, hand):
        if hand == self.player_hand:
            return 0
        else:
            for idx in range(self.num_opponents):
                if hand == self.opponents[idx].hand:
                    return idx+1
        return None

    def deal(self):
        """Deal the cards for the start of a game."""
        # print('deal():')
        print('=====>', 'Dealer is:', self.dealer, ':', self.getPlayerName(self.dealer))
        # deal_message = "Dealing cards..."
        deal_message =\
            "You are dealing..." if self.dealer == 0\
            else f"{self.opponents[self.dealer-1].info.name} is dealing..."
        # print(deal_message)
        self.show_status_notification(deal_message)
        # num_cards = 7 if self.num_opponents == 1 else 5
        num_cards = 5
        next_player = self.player_after(self.dealer)
        deal_state = DealState(cards_per_player=num_cards, round=0, next_player=next_player)
        self.deal_next_card(deal_state)

    def deal_next_card(self, deal_state: DealState):
        # print('deal_next_card')
        """Deal the next card, in the middle of a deal."""
        if deal_state.round >= deal_state.cards_per_player:
            self.on_deal_done()
            return

        card_ = self.stockpile.pop_top_card()
        if card_ is None:
            raise Exception('Not enough cards.')

        to_hand = self.player_hand\
                    if deal_state.next_player == 0\
                    else self.opponents[deal_state.next_player-1].hand

        def on_card_dealt(holder_):
            # Finish dealing the current card.
            back_side_up = to_hand != self.player_hand
            holder_.move_all_cards(to_hand, back_side_up)
            
            # Prepare next card to deal.
            round_ = deal_state.round+1\
                if deal_state.next_player == self.dealer\
                else deal_state.round
            next_player = self.player_after(deal_state.next_player)
            next_deal_state = DealState(deal_state.cards_per_player, round_, next_player)
            self.deal_next_card(next_deal_state)

        self.animate_cards([card_], to_hand.next_card_pos, on_complete=on_card_dealt, speed=self.move_speed)

    def on_deal_done(self):
        # print('on_deal_done')
        self.discard_top_card_from_stockpile()

    def setCurrentState(self, newState):
        '''
        Set the current state by applying gate operations.
        '''
        # if game_start:
        #     if state == "|11>":
        #         self.qc.x(0)
        #         self.qc.x(1)
        #     elif state == "|10>":
        #         self.qc.x(1)
        #     elif state == "|01>":
        #         self.qc.x(0)
        # else: # this should never happen once the game is ready
        #     raise
        currentState = self.getCurrentState()

        if currentState=="|00>":

            if newState=="|11>":
                self.qc.x(0)
                self.qc.x(1)
            elif newState=="|10>":
                self.qc.x(1)
            elif newState=="|01>":
                self.qc.x(0)

        elif currentState=="|01>":

            if newState=="|11>":
                self.qc.x(1)
            elif newState=="|10>":
                self.qc.x(0)
                self.qc.x(1)
            elif newState=="|00>":
                self.qc.x(0)

        elif currentState=="|10>":

            if newState=="|11>":
                self.qc.x(0)
            elif newState=="|01>":
                self.qc.x(0)
                self.qc.x(1)
            elif newState=="|00>":
                self.qc.x(1)

        elif currentState=="|11>":

            if newState=="|10>":
                self.qc.x(0)
            elif newState=="|00>":
                self.qc.x(0)
                self.qc.x(1)
            elif newState=="|01>":
                self.qc.x(1)

        self.show_current_state()

    def discard_top_card_from_stockpile(self):
        # print('discard_top_card_from_stockpile')
        def on_card_moved(holder_):
            raise # disabled
            card_ = holder_.pop_top_card()
            card_.back_up = False
            self.discard.add_card(card_, on_top=True)
            self.setCurrentState(card_.getState())
            self.next_turn()

        ## never start the discard pile with a gate
        # force_value = "Draw Two"
        # force_value = "Skip"
        # force_value = "Reverse"
        # force_value = ["Draw Two", "Skip", "Reverse"]
        # while (self.stockpile.cards[-1].getState()=='Gate') or (self.stockpile.cards[-1].getValue() not in force_value):
        #     self.stockpile.shuffle()
        while (self.stockpile.cards[-1].getState()=='Gate'):
            self.stockpile.shuffle()

        card_ = self.stockpile.pop_top_card()
        # play the first card normally if it is an action card
        self.lockActions()
        self.play_card(card_, on_complete=self.unlockActions, game_start=True)  # play the first card normally if it is an action card
        # self.animate_cards([card_], self.discard.next_card_pos, on_complete=on_card_moved, speed=self.move_speed)
    def unlockActions(self):
        self.action_lock = False
    def lockActions(self):
        self.action_lock = True

    def next_turn(self, players_to_skip=0):
        '''
        player 0 : human player
        players 1-N: computer players
        '''
        # print('next_turn called!')
        if self.turn == -1: # on first turn after dealing cards
            self.turn = self.player_after(self.dealer)
        else: # general next turn system
            self.turn = self.player_after(self.turn)
        for i in range(players_to_skip):
            turn_message = \
                "Skipping you..." if self.is_player_turn() \
                    else f"Skipping {self.opponents[self.turn - 1].info.name}..."
            self.show_status_notification(turn_message)
            print(turn_message)
            # pygame.time.wait(1000)
            self.turn = self.player_after(self.turn)

        # clear any leftover messages
        self.clear_dialog_title()

        # self.turn =\
        #     self.player_after(self.dealer) if self.turn == -1\
        #     else self.player_after(self.turn)
        turn_message =\
            "Your turn" if self.is_player_turn()\
            else f"{self.opponents[self.turn-1].info.name}'s turn"
        self.show_status_notification(turn_message)
        print('=====>', self.getCurrentPlayerInfoString())

    def is_player_turn(self):
        return self.turn == 0

    def can_play_card(self, card_):
        """Whether it's legal to play the given card at this time."""
        top_card = self.discard.cards[-1]
        if card_.getState() == "Gate":
            return True
        if card_.getState() == self.getCurrentState(): #top_card.getState():
            return True
        if card_.getValue() == top_card.getValue():
            return True
        return False

        # state = card_.suit
        # value = card_.rank
        # return QNO.canPlay(state, value, playerHand)
        # return True
        # top_card = self.discard.cards[-1]
        # return \
        #     card_.rank == 8\
        #     or (top_card.rank == 8 and card_.suit == self.chosen_suit)\
        #     or (top_card.rank != 8 and card_.rank == top_card.rank)\
        #     or (top_card.rank != 8 and card_.suit == top_card.suit)

    def player_after(self, player_idx):
        '''Return the index of the next player, but without actually changing to him. Takes into account play direction.'''
        if self.getPlayDirection()==0:
            return (player_idx + 1) % (self.num_opponents + 1)
        else:
            return (player_idx - 1) % (self.num_opponents + 1)

    def execute_game(self):
        # print('execute_game:')
        # self.show_current_state('TEST')
        for idx in range(0, self.num_opponents):
            # print('execute_game:idx:',idx)
            self.opponent_execute(idx)

    def opponent_execute(self, idx):
        '''
        N(opponents) = self.num_opponents
        N(players) = self.num_opponents + 1
        Opponents are indexed from 0 to N-1 for N opponents.
        Turns start with human player at turn=0, then go through turn=1...N for N opponents.
        So in general, current player idx = turn, current PC player idx = turn-1.
        '''
        if self.turn == idx+1:
            # It's this opponent's turn.
            if not self.action_lock:
                self.debug_info = 'opponent_execute called'
                self.action_lock = True
                def on_end_delay(): self.opponent_play(idx)
                self.opponent_delay(on_end_delay)
                self.debug_enabled = True
            else:
                if self.debug_enabled:
                    # print(f"Opponent's turn, but self.action_lock={self.action_lock}")
                    self.debug_enabled = False
        else:
            # player's turn
            # self.debug_enabled = True
            pass

    def opponent_play(self, idx):
        '''
        Main handler for the computer player.
        Called in a loop as long as it is that player's turn.
        '''
        opponent_ = self.opponents[idx]
        hand = opponent_.hand

        # print('opponent_play: idx:',idx,
        #       ', turn:', self.turn,
        #       ', name:', opponent_.info.name,
        #       ', self.num_opponents:', self.num_opponents,
        #       ', Ncards(player):', len(hand.cards),
        #       ', Ncards(deck):', len(self.stockpile.cards))

        # print(self.getCurrentPlayerInfoString())

        if self.must_choose_suit:
            possible_states = self.getPossibleStates()
            new_suit = opponent_.info.choose_suit(hand.cards, possible_states)
            # if self.getCurrentValue()=="H": # in case we use an H gate
            if len(possible_states)>2: # Happens in case of the H gate
                tmp = new_suit
                if new_suit in [0, 1]:
                    new_suit = 0
                else:
                    new_suit = 1
                print(f'H Gate: Converting chosen index: {tmp} -> {new_suit}')
            self.on_choose_suit(new_suit) # calls unlockActions() at the end
        elif self.getCurrentStateLength()!=1:
            # measurement required
            msg = "Performing measurement..."
            self.show_dialog_title(msg)
            print(msg)

            # get possible outcomes
            possible_measurements = self.getPossibleMeasurements()
            print('possible_measurements', possible_measurements)

            prediction = random.choice(possible_measurements)
            self.measure(prediction)
        else:
            def on_complete():
                if not hand.any_cards:
                    self.game_over(idx+1)
                else:
                    self.action_lock = False
            top_discard = self.discard.cards[-1]
            card_ = opponent_.info.try_select_card(hand.cards, top_discard, self.getCurrentState())
            if card_ is not None:
                card_ = hand.try_grab_card(card_)[0]
                self.play_card(card_, on_complete)
            else:
                if (not self.option_free_drawing) and self.hasPlayableCards(hand):
                    msg = f"{opponent_.info.name} has playable cards!"
                    self.show_player_prompt(msg)
                    for c in hand.cards:
                        print(c, self.can_play_card(c))
                    raise Exception(f'{msg} But still tried to draw cards!')
                self.draw_card_from_stockpile(on_complete)

    def test_getPossibleStates(self):
        for gate in ["X", "CNOT", "H"]:
            print(f'==> gate: {gate}')
            for q1 in range(2):
                for q0 in range(2):
                    print(f'|{q1}{q0}> -> {self.getPossibleStates(current_gate=gate, current_state=[q1,q0])}')

    def convertStateTupleToString(self, t):
        return f'|{t[0]}{t[1]}>'

    def getPossibleStates(self, current_gate=None, current_state=None):
        '''Return possible states after using gate.'''
        if current_gate is None:
            current_gate = self.getCurrentValue()
        if current_state is None:
            current_state = self.getCurrentState(as_tuple=True)

        if current_gate=="X":
            possible_states = []
            for idx in [0,1]:
                new_t = [current_state[0], current_state[1]]
                new_t[idx] = (new_t[idx]+1)%2
                possible_states.append( self.convertStateTupleToString(new_t) )
            return possible_states
        elif current_gate=="CNOT":
            possible_states = []
            for control_idx in [0,1]:
                new_t = [current_state[0], current_state[1]]
                if new_t[control_idx]==1:
                    target_idx = (control_idx+1)%2
                    new_t[target_idx] = (new_t[target_idx]+1)%2
                possible_states.append( self.convertStateTupleToString(new_t) )
            return possible_states
        elif current_gate=="H":
            possible_states = []
            for idx in [0, 1]:
                new_t = [current_state[0], current_state[1]]
                for val in [0, 1]: # possible random result
                    new_t[idx] = val
                    possible_states.append( self.convertStateTupleToString(new_t) )
            return possible_states

        raise # error by default
        return

    def drawCardsForPlayer(self, num_cards, player_idx, on_complete=None):
        """Deal **num_cards** cards to player **player_idx**."""
        print(f'Drawing {num_cards} for player {player_idx}: {self.getPlayerName(player_idx)}.')
        to_hand = self.player_hand if player_idx == 0 else self.opponents[player_idx - 1].hand

        message =\
            f"You draw {num_cards}." if player_idx == 0\
            else f"{self.opponents[player_idx-1].info.name}  draws {num_cards}."
        self.show_status_notification(message)
        deal_state = DealState(cards_per_player=num_cards, round=0, next_player=player_idx)
        self.draw_next_card(deal_state, on_complete=on_complete)
        return

    def cardsLeftInDeck(self):
        return len(self.stockpile.cards)
    def draw_next_card(self, deal_state: DealState, on_complete=None):
        """Draw the next card, in the middle of a card draw."""
        if deal_state.round >= deal_state.cards_per_player:
            # print('Drawing cards done.')
            if on_complete is not None:
                on_complete()
            else:
                print('on_complete not defined.')
                raise
            return
        # print(f'draw_next_card, cardsLeftInDeck: {self.cardsLeftInDeck()}')

        if not self.stockpile.any_cards:
            self.repopulate_stockpile(None)

        if not self.stockpile.any_cards:
            print('No cards left in stockpile.')
            return

        card_ = self.stockpile.pop_top_card()

        to_hand = self.player_hand \
            if deal_state.next_player == 0 \
            else self.opponents[deal_state.next_player - 1].hand

        def on_card_dealt(holder_):
            # Finish dealing the current card.
            back_side_up = to_hand != self.player_hand
            for i in holder_.cards:
                player_idx = self.getPlayerIndexFromHand(to_hand)
                print(f'drawCardsForPlayer: Player {player_idx} ({self.getPlayerName(player_idx)}) drew {i}')
            holder_.move_all_cards(to_hand, back_side_up)

            # Prepare next card to deal.
            round_ = deal_state.round + 1
            next_player = deal_state.next_player
            next_deal_state = DealState(deal_state.cards_per_player, round_, next_player)
            self.draw_next_card(next_deal_state, on_complete=on_complete)

        self.animate_cards([card_], to_hand.next_card_pos, on_complete=on_card_dealt, speed=self.draw_speed)

    def after_playing_card(self, card_, players_to_skip=0, on_complete=None, game_start=False):
        self.chosen_suit = None
        self.clear_dialog_title()
        self.setCurrentState(card_.getState())
        self.next_turn(players_to_skip=players_to_skip)

        if not self.stockpile.any_cards:
            self.repopulate_stockpile(on_complete)
        else:
            if on_complete is not None: on_complete()

    def play_card(self, card_, on_complete=None, game_start=False):
        def on_card_played(holder_):
            card_ = holder_.pop_top_card()
            card_.back_up = False
            self.discard.add_card(card_, on_top=True)

            if game_start:
                print('game_start: Card played is:', card_)
                # print('game_start: Current turn:', self.turn)
                self.turn = self.dealer
                # print('game_start: Current turn:', self.turn)
            else:
                print('Card played is:', card_)
            # print(card_.printInfos())

            players_to_skip = 0

            if card_.getValue() == "Reverse":
                # self.prompt_choose_suit()
                self.play_direction_image.toggleDirection()
                # if game_start:
                #     self.next_turn()
                #     print('Dealer was:', self.dealer)
                #     print('First player is:', self.turn)
                self.after_playing_card(card_, players_to_skip=0, on_complete=on_complete, game_start=game_start)
            elif card_.getValue() == "Skip":
                # print('SKIPPING!!!!')
                self.after_playing_card(card_, players_to_skip=1, on_complete=on_complete, game_start=game_start)
            elif card_.getValue() == "Draw Two":
                print('Draw Two!!!!')
                def afterCardsDrawn():
                    self.after_playing_card(card_, players_to_skip=0, on_complete=on_complete, game_start=game_start)

                player_idx = self.player_after(self.turn)
                self.drawCardsForPlayer(2, player_idx, on_complete=afterCardsDrawn)
            # elif card_.getState()=="Gate":
            elif card_.getValue()=="X" or card_.getValue()=="CNOT":
                print(f'{card_.getValue()} GATE PLAYED!!!!!')
                self.start_bgcolor_pulse_animation_for_8()
                self.prompt_choose_suit()
                if on_complete is not None:
                    # print('on_complete() called from play_card:on_card_played')
                    on_complete()
            elif card_.getValue()=="H":
                print(f'{card_.getValue()} GATE PLAYED!!!!!')
                self.start_bgcolor_pulse_animation_for_8()
                self.prompt_choose_suit()
                if on_complete is not None:
                    # print('H gate: on_complete() called from play_card:on_card_played')
                    on_complete()
                # self.after_playing_card(card_, players_to_skip=0, on_complete=on_complete, game_start=game_start)
            # elif card_.getValue()=="CNOT":
            #     print('CNOT GATE PLAYED!!!!!')
                # self.after_playing_card(card_, players_to_skip=0, on_complete=on_complete, game_start=game_start)
            else:
                self.after_playing_card(card_, players_to_skip=0, on_complete=on_complete, game_start=game_start)

            # if not self.stockpile.any_cards:
            #     self.repopulate_stockpile(on_complete)
            # else:
            #     if on_complete is not None: on_complete()

        # if card_.rank == 8:
        #     self.start_bgcolor_pulse_animation_for_8()

        card_.back_up = False
        self.animate_cards([card_], self.discard.next_card_pos, speed=self.move_speed,
            plotter_fn=self.get_plotter_fn_for_card(card_),
            on_complete=on_card_played)

    def get_plotter_fn_for_card(self, card_):
        return None  # Default plotter.
        # if card_.rank == 8:
        #     total_duration_ms = self.settings_json["crazy_8_spiral"]["duration_ms"]
        #     line_duration_ms = self.settings_json["crazy_8_spiral"]["line_duration_ms"]
        #     spiral_radius = self.settings_json["crazy_8_spiral"]["spiral_radius"]
        #     revolutions = self.settings_json["crazy_8_spiral"]["revolutions"]
        #     is_clockwise = self.settings_json["crazy_8_spiral"]["is_clockwise"]
        #
        #     def plotter_fn(start_pos, end_pos, _):
        #         return animation.LinearToSpiralPlotter(start_pos, end_pos,
        #                 total_duration_ms, line_duration_ms, spiral_radius, revolutions,
        #                 is_clockwise)
        #     return plotter_fn
        # else:
        #     return None # Default plotter.

    def start_bgcolor_pulse_animation_for_8(self):
        """Start pulsing the background for playing a 8."""
        color1 = self.settings_json["window"]["background_color"]
        color2 = [180+randint(0,50), 180+randint(0,50), 180+randint(0,50)]
        period_ms = self.settings_json["gui"]["play_8_background"]["period_ms"]

        self.bg_pulse_animation = self.create_bgcolor_pulse_animation(color1, color2, period_ms)
        self.add_animation(self.bg_pulse_animation)

    def prompt_choose_suit(self):
        # print('prompt_choose_suit called. Current player:', self.getCurrentPlayerInfoString())
        print('Current state:', self.getCurrentState(), ', Current gate:', self.getCurrentValue() ,' -> Possible states:', self.getPossibleStates())

        self.must_choose_suit = True
        if self.is_player_turn():
            self.show_choose_suit_dialog()
        else:
            self.show_dialog_title("Choosing qbit...")

    def getPlayerName(self, turn_idx):
        if turn_idx < 0:
            player_name = None
        elif turn_idx == 0:
            player_name = "you"
        else:
            opponent_ = self.opponents[turn_idx - 1]
            player_name = opponent_.info.name
        return player_name

    def getCurrentPlayerInfoString(self):
        if self.turn < 0:
            player_name = None
            Ncards = None
        elif self.turn == 0:
            hand = self.player_hand
            player_name = "you"
            Ncards = len(hand.cards)
        else:
            opponent_ = self.opponents[self.turn-1]
            hand = opponent_.hand
            player_name = opponent_.info.name
            Ncards = len(hand.cards)
        s = f'Current player: name: {player_name}, turn: {self.turn}, self.num_opponents:, {self.num_opponents},' \
            f' Ncards(player): {Ncards}, Ncards(deck): {len(self.stockpile.cards)}, self.must_choose_suit: {self.must_choose_suit}'
        return s

    def setPromptMessage(self, msg):
        self.prompt_message = msg

    def getCurrentValue(self):
        top_card = self.discard.cards[-1]
        return top_card.getValue()

    def show_choose_suit_dialog(self):
        """Show dialog asking player to choose which suit they want the next
        player to have to play.
        """
        top_card = self.discard.cards[-1]
        if top_card.getValue() == "X":
            msg = "X Gate played: Please specify qbit to act on:"
        elif top_card.getValue() == "H":
            msg = "H Gate played: Please specify qbit to act on:"
        elif top_card.getValue() == "CNOT":
            msg = "CNOT Gate played: Please specify control qbit:"
        else:
            raise
        self.show_dialog_title(msg)
        pos = self.settings_json["gui"]["dialog"]["content"]["position"]
        button_size = self.settings_json["gui"]["dialog"]["content"]["button_size"]
        x_base = pos[0]
        y = pos[1]
        button_margin = self.settings_json["gui"]["dialog"]["content"]["button_margin"]

        def choose(new_suit):
            def on_choose():
                self.on_choose_suit(new_suit)
            return on_choose

        for i in range(0, len(suit_info)):
            button_x = x_base + (button_size[0] * i) + (button_margin * i)
            button_pos = (button_x, y)
            button_rect = [*button_pos, *button_size]
            caption = suit_info[i].symbol
            self.gui_interface.show_button(button_rect, choose(i), f"{caption}",
                                            id_=f"suit_button{i}")

    def hide_choose_suit_dialog(self):
        # print('hide_choose_suit_dialog: self.is_choose_suit_dialog_hidden()', self.is_choose_suit_dialog_hidden())
        self.clear_dialog_title()
        for i in range(0, len(suit_info)):
            self.gui_interface.hide_by_id(f"suit_button{i}")
        # print('hide_choose_suit_dialog: self.is_choose_suit_dialog_hidden()', self.is_choose_suit_dialog_hidden())

    def is_choose_suit_dialog_hidden(self):
        for i in range(0, len(suit_info)):
            if self.gui_interface.has_id(f"suit_button{i}"):
                return False
        return True

    def show_measurement_dialog(self):
        """
        Show dialog asking player to guess the measurement outcome.
        """
        print('show_measurement_dialog: self.is_choose_suit_dialog_hidden()', self.is_choose_suit_dialog_hidden())

        msg = "Guess the measurement outcome:"
        self.show_dialog_title(msg)
        pos = self.settings_json["gui"]["dialog"]["content"]["position"]
        button_size = self.settings_json["gui"]["dialog"]["content"]["button_size"]
        x_base = pos[0]
        y = pos[1] + 50 # offset hack to avoid direct activation after finishing selecting qbit
        button_margin = self.settings_json["gui"]["dialog"]["content"]["button_margin"]

        possible_measurements = self.getPossibleMeasurements()
        print('possible_measurements', possible_measurements)

        def predict(prediction):
            def on_predict():
                self.measure(prediction)
            return on_predict

        for i in range(self.getCurrentStateLength()):
            button_x = x_base + (button_size[0] * i) + (button_margin * i)
            button_pos = (button_x, y)
            button_rect = [*button_pos, *button_size]
            caption = possible_measurements[i]
            self.gui_interface.show_button(button_rect, predict(caption), f"{caption}", id_=f"prediction_button{i}")

    def hide_measurement_dialog(self):
        self.clear_dialog_title()
        self.gui_interface.hide_by_id(f"prediction_button{0}")
        self.gui_interface.hide_by_id(f"prediction_button{1}")

    def measure(self, prediction):
        old_state = self.getCurrentState(allow_superposition=True)

        ##### perform measurement
        self.qc.measure_all() # Adds measurement to all qubits.
        job = execute(self.qc, Aer.get_backend('qasm_simulator'), shots=1) # perform measurement
        # we can get the result of the outcome as follows
        counts = job.result().get_counts(self.qc)
        # print(counts)  # counts is a dictionary
        if len(counts.keys()) > 1:
            raise
        result = list(counts.keys())[0] # result in 'ab' form
        # print('result', f'|{result}>')

        ##### re-initialize quantum circuit with measured value, so we can use Statevector again.
        self.qc = QuantumCircuit(2)
        self.qc.initialize(result, self.qc.qubits)

        statevector = Statevector(self.qc)  # works
        # print(statevector.probabilities_dict())

        new_state = self.getCurrentState()
        msg = f'Performed measurement: {old_state} -> {new_state}. The guess was: {prediction}. Guess correct: {prediction==new_state}'
        print(msg)

        self.hide_measurement_dialog()
        self.show_current_state()

        if prediction==new_state:
            if self.turn==0:
                msg = 'Your guess was correct! :)'
            else:
                msg = f"{self.getPlayerName(self.turn)}'s guess was correct! :)"
        else:
            if self.turn==0:
                msg = 'Your guess was not correct! :('
            else:
                msg = f"{self.getPlayerName(self.turn)}'s guess was not correct! :("
        self.show_dialog_title(msg)
        print(msg)

        # Release lock after measurement is done.
        def on_complete():
            self.bg_pulse_animation.is_completed = True
            self.next_turn()
            self.action_lock = False

        #  If the player guesses correctly, then the next player draws four cards.
        #  If the player guesses incorrectly, then the player must draw a single card.
        if prediction==new_state:
            self.drawCardsForPlayer(4, self.player_after(self.turn), on_complete=on_complete)
            # player_idx = self.player_after(self.turn)
            # self.drawCardsForPlayer(2, player_idx, on_complete=afterCardsDrawn)
        else:
            self.drawCardsForPlayer(1, self.turn, on_complete=on_complete)

    def on_choose_suit(self, new_suit):
        '''
        Used by human and PC player after choice has been made.
        '''
        self.chosen_suit = new_suit
        self.must_choose_suit = False
        self.hide_choose_suit_dialog()
        suit = suit_info[new_suit]
        old_state = self.getCurrentState()
        msg = f"{suit.description} was chosen."
        self.show_dialog_title(msg)
        print(msg)

        self.measurement_required = False

        # apply gate
        top_card = self.discard.cards[-1]
        if top_card.getValue() == "X":
            msg = f"X Gate played: Applying qc.x({suit.qiskit_index})"
            print(msg)
            self.qc.x(suit.qiskit_index)
        elif top_card.getValue() == "H":
            msg = f"H Gate played: Applying qc.h({suit.qiskit_index})"
            print(msg)
            self.qc.h(suit.qiskit_index)
            self.measurement_required = True
        elif top_card.getValue() == "CNOT":
            c = suit.qiskit_index
            t = (suit.qiskit_index+1)%2
            msg = f"CNOT Gate played: Applying qc.cx({c}, {t})"
            print(msg)
            self.qc.cx(c, t)
        else:
            raise

        print('state change:', old_state, '->', self.getCurrentState(allow_superposition=True))
        self.show_current_state() # Update state information display.

        if not self.measurement_required:
            self.bg_pulse_animation.is_completed = True
            self.next_turn()
            self.action_lock = False
        else:
            if self.turn==0: # human player
                self.show_measurement_dialog()
            else: # computer player
                self.action_lock = False # further processing will be done in the main handler to mirror behaviour of the human handling

    def draw_card_from_stockpile(self, on_complete=None, player_idx=None):
        if player_idx is None:
            hand = self.player_hand if self.is_player_turn() else self.opponents[self.turn - 1].hand
            back_up = not self.is_player_turn()
        else:
            if player_idx == 0:
                hand = self.player_hand
                back_up = False
            else:
                hand = self.opponents[player_idx - 1].hand
                back_up = True

        def on_card_drawn(holder_):
            card_ = holder_.pop_top_card()
            card_.back_up = back_up
            hand.add_card(card_, on_top=True)
            print(f'draw_card_from_stockpile: Player {self.turn} drew {card_}')
            if not self.stockpile.any_cards:
                self.repopulate_stockpile(on_complete)
            else:
                if on_complete is not None: on_complete()

        card_ = self.stockpile.pop_top_card()
        self.animate_cards([card_], hand.next_card_pos, on_complete=on_card_drawn, speed=self.move_speed)

    def getScoreBoard(self):
        # get scores
        score = {}
        score['you'] = len(self.player_hand.cards)
        for idx in range(self.num_opponents):
            score[self.opponents[idx].info.name] = len(self.opponents[idx].hand.cards)

        # turn into a sorted list, with support for ties
        score_board = process_score(score)
        return score_board


    def end_game(self):
        self.game_over()

        # score_board = self.getScoreBoard()
        # # print to terminal
        # printScoreBoard(score_board)
        #
        # winner_list = score_board[0][1]
        # if didPlayerWin('you', score_board):
        #     self.game_over(0, winner_list=winner_list, cards_left=score_board[0][0])
        # else:
        #     self.game_over(1, winner_list=winner_list, cards_left=score_board[0][0])

        # score_sorted = {k: v for k, v in sorted(score.items(), key=lambda item: item[1])}
        # for k,v in score_sorted.items():
        #     print(k,v)
        return

    def repopulate_stockpile(self, on_complete):
        if not self.option_replenish_stack:
            self.end_game()
            return
        self.repopulate_stockpile_card(on_complete)

    def repopulate_stockpile_card(self, on_complete):
        if len(self.discard.cards) <= 1:
            self.stockpile.shuffle()
            if on_complete is not None: on_complete()
        else:
            def on_card_move(holder_):
                card_ = holder_.pop_top_card()
                self.stockpile.add_card(card_)
                self.repopulate_stockpile_card(on_complete)

            card_ = self.discard.pop_bottom_card()
            card_.back_up = True
            self.animate_cards([card_], self.stockpile.next_card_pos, on_complete=on_card_move, speed=self.move_speed)

    def opponent_delay(self, on_delay_complete):
        """Simulate AI opponent waiting before making its move."""
        delay_ms = randint(self.opponent_min_delay_ms, self.opponent_max_delay_ms)
        threading.Timer(delay_ms / 1000, on_delay_complete).start()

    def game_over(self, winner_idx=None):
        # print game end
        print('=====>', 'GAME OVER!')
        # stop all actions
        self.lockActions()
        # clear other dialogs
        self.clear_dialog_title()
        # get scores
        score_board = self.getScoreBoard()
        # print to terminal
        printScoreBoard(score_board)

        # relevant infos
        winner_list = score_board[0][1]
        cards_left = score_board[0][0]

        # determine if human player won
        if winner_idx is None:
            if didPlayerWin('you', score_board):
                winner_idx = 0
            else:
                winner_idx = 1

        # play sounds
        if winner_idx == 0:
            self.playWinningSound()
        else:
            self.playLosingSound()

        # animate background
        self.start_game_over_bgcolor_pulse_animation()

        # create message
        if len(winner_list)==1 and cards_left==0:
            message = "You win!!!" if winner_idx == 0 else f"{self.opponents[winner_idx-1].info.name} wins!!!"
        else:
            message = f"Winner(s) with {cards_left} cards left: {', '.join(winner_list)}"

        # display message
        self.show_status_notification(f"GAME OVER! {message}", color=(255, 100, 100), message_json=self.settings_json["gui"]["message_game_over"])
        print(message)

    def start_game_over_bgcolor_pulse_animation(self):
        """Start pulsing the background for game over."""
        color1 = self.settings_json["gui"]["game_over_background"]["color1"]
        color2 = self.settings_json["gui"]["game_over_background"]["color2"]
        period_ms = self.settings_json["gui"]["game_over_background"]["period_ms"]

        self.bg_pulse_animation = self.create_bgcolor_pulse_animation(color1, color2, period_ms)
        self.add_animation(self.bg_pulse_animation)

    def clear_opponents(self):
        for i in range(0, len(self.opponents)):
            self.opponents[i].hand.move_all_cards(self.stockpile)
            self.remove_rendered_object(self.opponents[i].hand)
            self.gui_interface.hide_by_id(f"OPPONENT_{i}_NAME")

        self.num_opponents = 0
        self.opponents = []

    def show_message(self, text, message_json, message_id, color):
        pos = message_json["position"]
        size = message_json["text_size"]
        color = color or message_json["text_color"]
        self.gui_interface.hide_by_id(message_id)
        self.gui_interface.show_label(position=pos, text=text, text_size=size,
                                      timeout=0, color=color, id_=message_id)

    def clear_status_notification(self):
        self.gui_interface.hide_by_id(STATUS_NOTIFICATION)

    def show_status_notification(self, text, color=None, message_json=None):
        # print('show_status_notification')
        if message_json is None:
            message_json = self.settings_json["gui"]["message"]
        self.show_message(text, message_json, STATUS_NOTIFICATION, color)

    def clear_dialog_title(self):
        self.gui_interface.hide_by_id(DIALOG_TITLE)

    def show_dialog_title(self, text, color=None):
        self.show_message(text, self.settings_json["gui"]["dialog"]["title"], DIALOG_TITLE, color)

    def getCurrentStateLength(self):
        statevector = Statevector(self.qc)
        prob_dict = statevector.probabilities_dict()
        return len(prob_dict)

    def getPossibleMeasurements(self):
        '''Return the possible measurment results.'''
        statevector = Statevector(self.qc)
        prob_dict = statevector.probabilities_dict()
        L = [f'|{k}>' for k in prob_dict.keys()]
        return L

    def getCurrentState(self, as_tuple=False, allow_superposition=False):
        statevector = Statevector(self.qc)
        # print(statevector)
        # print(statevector.probabilities())
        # print(statevector.probabilities_dict())
        prob_dict = statevector.probabilities_dict()

        # special handling of superposition
        if len(prob_dict) != 1:
            if (not allow_superposition) or as_tuple:
                raise Exception(f'More than one possible value: prob_dict={prob_dict}')
            else:
                L = [f'|{k}>' for k in prob_dict.keys()]
                s = '+'.join(L)
                return s

        s = f'|{list(prob_dict.keys())[0]}>'
        if as_tuple:
            return [int(s[1]), int(s[2])]
        else:
            return s

    def show_current_state(self, color=None):
        text = f'Current state: {self.getCurrentState(allow_superposition=True)}'
        text += f', Turn direction: {self.play_direction_image.playDirection}'
        print(text)
        self.show_message(text, self.settings_json["gui"]["current_state"], CURRENT_STATE, color)
        # pos = self.settings_json["gui"]["current_state"]
        # self.show_message(text, , "TURN_DIRECTION", color)

    def clear_player_prompt(self):
        self.gui_interface.hide_by_id(PLAYER_PROMPT)

    def show_player_prompt(self, text, color=None):
        self.show_message(text, self.settings_json["gui"]["player_prompt"], PLAYER_PROMPT, color)

    def process_mouse_event(self, pos, down, double_click):
        """ Put code that handles mouse events here. For example: grab card from a deck on mouse
            down event, drop card to a pile on mouse up event etc.
            This method is called every time mouse event is detected.
            :param pos: tuple with mouse coordinates (x, y)
            :param down: boolean, True for mouse down event, False for mouse up event
            :param double_click: boolean, True if it's a double click event
        """
        #print(f'process_mouse_event pos=${pos}, down=${down}, dbl-click=${double_click}')
        if down:
            self.on_player_click(pos)
        else:
            self.clear_player_prompt()

    def on_player_click(self, pos):
        '''
        Main handler for the human player.
        '''
        if self.action_lock:
            # Some animation or other is in progress. No action allowed.
            return
        elif self.is_player_turn():
            if self.must_choose_suit:
                self.show_player_prompt("You played a gate and must choose a qbit!")
            elif self.getCurrentStateLength()!=1:
                self.show_player_prompt("The state is in a superposition. You must perform a measurement!")
            elif self.player_hand.is_clicked(pos):
                card_clicked, _ = self.player_hand.card_at(pos)
                if card_clicked is not None:
                    if self.can_play_card(card_clicked):
                        cards = self.player_hand.try_grab_card(card_clicked)
                        self.action_lock = True
                        def on_complete():
                            if not self.player_hand.any_cards:
                                self.game_over(0)
                            else:
                                self.action_lock = False
                        self.play_card(cards[0], on_complete)
                    else:
                        self.show_player_prompt("You can't play that card.")
            elif self.stockpile.is_clicked(pos):
                if (not self.option_free_drawing) and self.hasPlayableCards(self.player_hand):
                    self.show_player_prompt("You have playable cards!")
                    return
                self.action_lock = True
                def on_complete(): self.action_lock = False
                self.draw_card_from_stockpile(on_complete)
        else:
            self.show_player_prompt("It's not your turn!")

    def hasPlayableCards(self, hand):
        for card_ in hand.cards:
            if self.can_play_card(card_):
                return True
        return False

    def create_bgcolor_pulse_animation(self, color1, color2, period_ms):
        """Set up animation to pulse the background between given colors.
        :param color1 Tuple(int, int, int): First color to pulse between.
        :param color2 Tuple(int, int, int): Second color to pulse between.
        :param period_ms int: Pulse period in milliseconds.
        """

        def set_color(color):
            self.background_color = color

        def on_complete():
            self.background_color = None

        return animation.ColorPulseAnimation(color1, color2, period_ms, set_color, on_complete)


def main():
    # get the main script's path
    import os
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    
    # change into the script's directory (TODO: A better solution would be to load file relative to that path instead of using chdir.)
    os.chdir(dname)

    # Seed random number generator.
    seed(int(round(time.time() * 1000)))

    json_path = os.path.join(os.getcwd(), 'settings2.json')
    # json_path = os.path.join(dname, 'settings2.json')
    crazy8s_app = game_app.GameApp(json_path=json_path, controller_cls=Crazy8sController)
    crazy8s_app.execute()

def hack_json():
    with open('settings2.json','r') as json_file:
        json_dict = json.load(json_file)
        for k,v in json_dict.items():
            print(k,v)

def process_score(score):
    score_board = []

    unique_scores = set()
    for k,v in score.items():
        unique_scores.add(v)

    unique_scores = list(unique_scores)
    unique_scores.sort()
    # print(unique_scores)

    for s in unique_scores:
        player_list = []
        for k, v in score.items():
            if v == s:
                player_list.append(k)
        score_board.append( (s, player_list) )

    # score_sorted = {k: v for k, v in sorted(score.items(), key=lambda item: item[1])}
    # for k, v in score_sorted.items():
    #     print(k, v)
    return score_board

def printScoreBoard(score_board):
    for idx, e in enumerate(score_board):
        cards_left = e[0]
        player_list = e[1]
        player_str = ', '.join(player_list)
        print(f'{idx+1}) cards left: {cards_left}, player(s): {player_str}')

def didPlayerWin(name, score_board):
    return (name in score_board[0][1])

def test_score():
    score={}
    score['a']=100
    score['b']=1000
    score['c']=90
    score['d']=9
    score['e']=9
    score['f']=90

    score_board = process_score(score)
    printScoreBoard(score_board)
    for k in score.keys():
        print(k, didPlayerWin(k, score_board))

if __name__ == '__main__':
    # hack_json()
    main()
    # test_score()
