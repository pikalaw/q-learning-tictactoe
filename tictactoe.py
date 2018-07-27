from abc import ABC, abstractmethod
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


def match(player_x, player_o, stats):
  board = Board()

  players = {X: player_x, O: player_o}
  player_x.start(board, X)
  player_o.start(board, O)

  next_to_play = X

  while True:
    winner = board.winner
    if winner:
      print('Game won by {}.'.format(_NAMES[winner]), flush=True)
      stats[winner] += 1
      break
    if board.is_tied:
      print('Game is tied.', flush=True)
      stats[None] += 1
      break

    move = players[next_to_play].play()
    print('Player {} plays {}'.format(_NAMES[next_to_play], move))
    board.play(next_to_play, move)
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


class QLearningPlayer(Player):
  def __init__(self):
    pass

  def start(self, board, player):
    self.board = board
    self.me = player

  def play(self):
    return random.choice(self._best_moves[Game(self.board, self.me)])

  def end(self):
    pass


def empty_stats():
  return {X: 0, O: 0, None: 0}


def main():
  print('Computing best moves...', end=' ', flush=True)
  start_time = time.time()
  best_moves = find_best_moves()
  end_time = time.time()
  print('Done in {} seconds'.format(end_time - start_time), flush=True)

  player_o = PerfectPlayer(best_moves)
  player_x = HumanPlayer()

  stats = empty_stats()
  for i in range(100):
    print('Game #{}'.format(i))
    match(player_x, player_o, stats)
  print(stats)


if __name__ == '__main__':
  main()
