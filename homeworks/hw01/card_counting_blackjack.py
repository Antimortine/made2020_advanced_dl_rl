import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
single_deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4


def cmp(a, b):
    return float(a > b) - float(a < b)


def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]


class CardCountingBlackjackEnv(gym.Env):
    """Modification of BlackjackEnv which implements card counting and doubling.
    It uses Uston APC counting strategy with side count for Aces."""

    def __init__(self,
                 num_decks: int,
                 reshuffle_at: int,
                 natural: bool,
                 counting_strategy: str = 'uston_apc',
                 count_aces: bool = True):
        assert num_decks >= 1 and isinstance(num_decks, int)
        self.num_decks = num_decks
        assert counting_strategy in [
            'uston_apc',  # https://www.qfit.com/cardcounting/Uston-APC/
            'reko']  # https://www.qfit.com/cardcounting/REKO/
        self.counting_strategy = counting_strategy
        self.count_aces = count_aces
        self.action_space = spaces.Discrete(3)  # stick, hit, double
        self.observation_space = self._get_observation_space()
        self.seed()

        self.deck = single_deck * num_decks
        assert 15 <= reshuffle_at < len(self.deck) and isinstance(reshuffle_at, int)
        self.reshuffle_at = reshuffle_at
        self.next_card_idx = len(self.deck) - 1
        self.running_count = 0
        self.aces_count = 0
        self.shuffle_deck()

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural
        # self.reset()

    def _get_observation_space(self):
        sum_space = spaces.Discrete(32)  # Player's sum
        dealers_card_space = spaces.Discrete(11)  # Dealer's open card
        usable_ace_space = spaces.Discrete(2)  # Does player have usable ace?
        if self.counting_strategy == 'uston_apc':
            count_space = spaces.Box(low=-52 * self.num_decks,
                                     high=52 * self.num_decks,
                                     shape=(1,),
                                     dtype=np.int32)
        elif self.counting_strategy == 'reko':
            count_space = spaces.Box(low=-20 * self.num_decks,
                                     high=24 * self.num_decks,
                                     shape=(1,),
                                     dtype=np.int32)
        else:
            raise NotImplementedError
        aces_space = spaces.Discrete(self.num_decks * 4 + 1)  # Number of aces out
        if self.count_aces:
            return spaces.Tuple((
                sum_space,
                dealers_card_space,
                usable_ace_space,
                count_space,
                aces_space))
        else:
            return spaces.Tuple((
                sum_space,
                dealers_card_space,
                usable_ace_space,
                count_space))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def shuffle_deck(self):
        self.np_random.shuffle(self.deck)
        self.next_card_idx = len(self.deck) - 1
        self.running_count = 0
        self.aces_count = 0

    def update_counts(self, card):
        if self.counting_strategy == 'uston_apc':
            if card == 1:
                self.aces_count += 1
            elif card in [2, 8]:
                self.running_count += 1
            elif card in [3, 4, 6, 7]:
                self.running_count += 2
            elif card == 5:
                self.running_count += 5
            elif card == 9:
                self.running_count -= 1
            elif card == 10:
                self.running_count -= 3
            else:
                raise ValueError
        elif self.counting_strategy == 'reko':
            if card == 1:
                self.aces_count += 1
            if card in [1, 10]:
                self.running_count -= 1
            elif card in [2, 3, 4, 5, 6, 7]:
                self.running_count += 1
            elif card in [8, 9]:
                pass
            else:
                raise ValueError
        else:
            raise NotImplementedError

    def draw_card(self, hidden: bool = False):
        card = self.deck[self.next_card_idx]
        self.next_card_idx -= 1
        if not hidden:  # If it's hidden dealer's card, don't count
            self.update_counts(card)
        return card

    def draw_hand(self, hide_second: bool = False):
        return [self.draw_card(), self.draw_card(hide_second)]

    def step(self, action):
        assert self.action_space.contains(action)
        # double: add a card to players hand, check if bust, play out the dealers hand, and score with double reward
        if action == 2:
            done = True
            self.player.append(self.draw_card())
            if is_bust(self.player):
                reward = -2.
            else:
                while sum_hand(self.dealer) < 17:
                    self.dealer.append(self.draw_card())
                reward = cmp(score(self.player), score(self.dealer)) * 2
        elif action == 1:  # hit: add a card to players hand and return
            self.player.append(self.draw_card())
            if is_bust(self.player):
                done = True
                reward = -1.
            else:
                done = False
                reward = 0.
        else:  # stick: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(self.draw_card())
            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1.:
                reward = 1.5
        if done:
            self.update_counts(self.dealer[1])  # Now player can see initially hidden dealer's card
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        player_sum = sum_hand(self.player)
        if self.count_aces:
            return (
                player_sum,  # Player's sum
                self.dealer[0],  # Dealer's open card
                usable_ace(self.player),  # Does player have usable ace?
                self.running_count,  # Running count
                self.aces_count  # Number of aces out
            )
        else:
            return (
                player_sum,  # Player's sum
                self.dealer[0],  # Dealer's open card
                usable_ace(self.player),  # Does player have usable ace?
                self.running_count,  # Running count
            )

    def reset(self):
        if self.next_card_idx + 1 <= self.reshuffle_at:
            self.shuffle_deck()
        self.dealer = self.draw_hand(hide_second=True)
        self.player = self.draw_hand()
        return self._get_obs()
