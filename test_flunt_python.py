import collections

Card = collections.namedtuple('Card', ['rank', 'suit'])

class FrenchDeck(object):
    ranks = [str(n) for n in range(2, 11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()
    """docstring for FrenchDeck"""
    def __init__(self):
        self._cards = [Card(rank, suit) for suit in self.suits
            for rank in self.ranks]

    def __len__(self):
        return len(self._cards)

    def __getitem__(self, position):
        return self._cards[position]

if __name__ == "__main__":
    bear_card = Card['7', 'diamonds']
    print(bear_card)
    deck = FrenchDeck()
    print(len(deck))
    print(f'{deck[0]=}, {deck[-1]=}')