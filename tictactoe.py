from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
import random
import time


EMPTY = 0
X = 1
O = 2

_NAMES = {X:'X', O:'O', EMPTY:' ', None:'-'}


class Board(object):
  def __init__(self, board=None):
    if board:
      self._board = board
    else:
      self._board = [EMPTY]*9
    self._winner = None
    self._is_tied = False

  def __repr__(self):
    return 'Board({})'.format(self._board)

  @property
  def winner(self):
    if self._winner:
      return self._winner

    for row in range(3):
      if (self._board[row*3] != EMPTY and
          self._board[row*3] == self._board[row*3+1] == self._board[row*3+2]):
        self._winner = self._board[row*3]
        return self._winner

    for col in range(3):
      if (self._board[col] != EMPTY and
          self._board[col] == self._board[3+col] == self._board[6+col]):
        self._winner =  self._board[col]
        return self._winner

    if (self._board[4] != EMPTY and
        self._board[0] == self._board[4] == self._board[8]):
      self._winner =  self._board[4]
      return self._winner

    if (self._board[4] != EMPTY and
        self._board[2] == self._board[4] == self._board[6]):
      self._winner =  self._board[4]
      return self._winner

    return None

  @property
  def is_tied(self):
    if not self._is_tied:
      self._is_tied = ((not self.winner) and
          sum(1 for cell in self._board if cell in [X, O]) == 9)
    return self._is_tied

  @property
  def is_valid(self):
    for piece in self._board:
      assert piece in [X, O, EMPTY]
  
  @property
  def valid_moves(self):
    if self.winner:
      return []
    else:
      return [i for i in range(9) if self._board[i] == EMPTY]

  def __str__(self):
    return ''.join([
        '\n',
        '\n-----\n'.join([
            '|'.join([_NAMES[self._board[row*3+col]] for col in range(3)])
            for row in range(3)
        ]),
        '\n'
    ])

  def __hash__(self):
    return hash(repr(self))

  def __eq__(self, other):
    return repr(self) == repr(other)

  def play(self, player, move):
    assert 0 <= move <= 8
    assert self._board[move] == EMPTY
    self._board[move] = player


def _opponent(player):
  return {X:O, O:X}[player]


class Game(object):
  def __init__(self, board=None, next_player=X):
    if board:
      self._board = board
    else:
      self._board = Board()
    self._next_player = next_player

  def __repr__(self):
    return 'Game({}, {})'.format(repr(self.board), self.next_player)

  def __str__(self):
    return 'Player: {}{}'.format(_NAMES[self.next_player], self.board)

  @property
  def next_player(self):
    return self._next_player

  @property
  def board(self):
    return self._board

  def __hash__(self):
    return hash(repr(self))

  def __eq__(self, other):
    return repr(self) == repr(other)

  def play(self, move):
    assert self.next_player in [X, O]
    self.board.play(self.next_player, move)
    if self.board.is_tied or self.board.winner:
      self._next_player = None
    else:
      self._next_player = _opponent(self.next_player)


def find_best_moves():
  WIN = 1
  TIE = 0
  LOSE = -1

  best_moves = {}
  start_game = Game()

  def dfs(game):
    nonlocal best_moves
    valid_moves = game.board.valid_moves

    opponent_results = {}
    for move in valid_moves:
      try_game = deepcopy(game)
      try_game.play(move)
      if try_game.board.winner:
        opponent_results[move] = LOSE
      elif try_game.board.is_tied:
        opponent_results[move] = TIE
      else:
        opponent_results[move] = dfs(try_game)
    
    best_moves[game] = [
        move for move, opponent_result in opponent_results.items()
        if opponent_result == LOSE]
    if best_moves[game]:
      return WIN

    best_moves[game] = [
        move for move, opponent_result in opponent_results.items()
        if opponent_result == TIE]
    if best_moves[game]:
      return TIE

    return LOSE

  dfs(start_game)
  return best_moves


class Player(ABC):
  @abstractmethod
  def start(self, board, player):
    pass

  @abstractmethod
  def play(self):
    pass

  @abstractmethod
  def end(self):
    pass


def match(player_x, player_o, stats, output=True):
  board = Board()

  players = {X: player_x, O: player_o}
  player_x.start(board, X)
  player_o.start(board, O)

  next_to_play = X

  while True:
    winner = board.winner
    if winner:
      if output:
        print('Game won by {}.'.format(_NAMES[winner]), flush=True)
      stats[winner] += 1
      break
    if board.is_tied:
      if output:
        print('Game is tied.', flush=True)
      stats[None] += 1
      break

    move = players[next_to_play].play()
    if output:
      print('Player {} plays {}'.format(_NAMES[next_to_play], move))
    board.play(next_to_play, move)
    if output:
      print(board)
    next_to_play = _opponent(next_to_play)
     
  for player in players.values():
    player.end()


class RandomPlayer(Player):
  def start(self, board, player):
    self.board = board
    self.me = player

  def play(self):
    return random.choice(self.board.valid_moves)

  def end(self):
    pass


class PerfectPlayer(Player):
  def __init__(self, best_moves):
    self._best_moves = best_moves

  def start(self, board, player):
    self.board = board
    self.me = player

  def play(self):
    return random.choice(self._best_moves[Game(self.board, self.me)])

  def end(self):
    pass


class HumanPlayer(Player):
  def __init__(self):
    self._name = input('What is your name? ')

  @property
  def name(self):
    return self._name

  def start(self, board, player):
    self.board = board
    self.me = player
    print('{}! You are {}.'.format(self.name, _NAMES[player]))

  def play(self):
    valid_moves = self.board.valid_moves
    while True:
      print('{}!\n{}'.format(self.name, self.board))
      print('Valid moves are {}'.format(valid_moves))
      move = int(input('Your move: '))
      if move in valid_moves:
        return move
      else:
        print('ERROR: Invalid move!')

  def end(self):
    winner = self.board.winner
    if winner == self.me:
      print("{}! You've won the game!".format(self.name))
    elif winner:
      print("{}! You've lost the game!".format(self.name))
    else:
      print("{}! Tied game.".format(self.name))


def empty_q_score():
  # (state, action) -> expected total reward.
  return defaultdict(lambda: defaultdict(float))


def empty_observed_state_transition():
  # (state, action, next_state) -> number of times 'state' with 'action'
  # resulted in next_state.
  return defaultdict(lambda: defaultdict(lambda: defaultdict(int)))


class QLearningPlayer(Player):
  def __init__(self, q_score, observed_state_transition, learning_rate=1.0,
      discount_factor=1.0, e_greedy=0.0):
    self.q_score = q_score
    self.observed_state_transition = observed_state_transition
    self.learning_rate = learning_rate
    self.discount_factor = discount_factor
    self.e_greedy = e_greedy

  def start(self, board, player):
    self.board = board
    self.me = player
    self._frames = []

  def play(self):
    current_state = self._current_game_state
    current_moves = self.board.valid_moves
    selected_move = self._compute_best_move(current_state, current_moves)
    self._frames.append((current_state, selected_move))
    return selected_move

  def end(self):
    next_state = self._current_game_state
    for state, move in reversed(self._frames):
      self._update_observed_state_transition(state, move, next_state)
      self._update_q_score(state, move)
      next_state = state

  @property
  def _current_game_state(self):
    return QLearningPlayer._pack_game_state(self.board, self.me)

  @staticmethod
  def _pack_game_state(board, player):
    return repr(board), player
      
  @staticmethod
  def _unpack_game_state(state):
    board_repr, player = state
    return eval(board_repr), player
      
  def _compute_best_move(self, state, valid_moves):
    if random.random() < self.e_greedy:
      return random.choice(valid_moves)
    else:
      q_scores = self.q_score[state]
      # Make sure all moves are initialized.
      for move in valid_moves:
        q_scores[move]
      return max(q_scores, key=lambda move: q_scores[move])

  def _update_observed_state_transition(self, state, move, next_state):
    self.observed_state_transition[state][move][next_state] += 1

  def _update_q_score(self, state, move):
    if self.learning_rate == 0:
      # Won't learn. So, just skip the rest of the computation.
      return
    self.q_score[state][move] = (
        (1 - self.learning_rate) * self.q_score[state][move] +
        self.learning_rate * self._compute_new_q_score(state, move))

  def _compute_new_q_score(self, state, move):
    observed_next_states = self.observed_state_transition[state][move]
    assert len(observed_next_states) > 0

    new_q_score = 0

    total = sum(observed_next_states.values())
    for potential_next_state, count in observed_next_states.items():
      new_q_score += (
        count / total * (
          QLearningPlayer._reward(potential_next_state) +
          max(self.q_score[potential_next_state].values(), default=0)
        )
      )

    return new_q_score

  @staticmethod
  def _reward(state):
    board, player = QLearningPlayer._unpack_game_state(state)
    winner = board.winner
    if winner == player:
      return 100
    if winner:
      return -100
    if board.is_tied:
      return 50
    return -1

 
# Returns q_score, observed_state_transition.
def train_q_learner(against_player, as_x_or_o):
  q_score = empty_q_score()
  observed_state_transition = empty_observed_state_transition()

  training_q_learner = QLearningPlayer(q_score, observed_state_transition,
      learning_rate=1, discount_factor=1, e_greedy=1)

  players = {
    as_x_or_o: against_player,
    _opponent(as_x_or_o): training_q_learner
  }

  num_episodes = 1000
  stats = empty_stats()
  for i in range(num_episodes):
    match(players[X], players[O], stats, output=False)
    # Linear decay of e-greedy.
    training_q_learner.e_greedy = (num_episodes - i) / num_episodes
  print(stats)

  return q_score, observed_state_transition
  

def build_q_learned_player(q_score, observed_state_transition):
  return QLearningPlayer(q_score, observed_state_transition,
      learning_rate=0, discount_factor=1, e_greedy=0)


def empty_stats():
  return {X: 0, O: 0, None: 0}


@contextmanager
def time_this(task_description):
  print(task_description, '...', end=' ', flush=True)
  start_time = time.time()
  yield
  end_time = time.time()
  print('Done in {:.2f} seconds'.format(end_time - start_time), flush=True)
  

def main():
  with time_this('Computing best moves'):
    best_moves = find_best_moves()

  player_o = PerfectPlayer(best_moves)

  with time_this('Training Q-Learner'):
    q_score, observed_state_transition = train_q_learner(player_o, O)

  player_x = build_q_learned_player(q_score, observed_state_transition)

  num_games = 100
  stats = empty_stats()
  for i in range(num_games):
    print('Game #{} out of {}'.format(i, num_games))
    match(player_x, player_o, stats, output=True)
  print(stats)


if __name__ == '__main__':
  main()
