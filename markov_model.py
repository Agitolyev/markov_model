from __future__ import annotations

from typing import Dict, List
import argparse
import stdrandom
import ctypes


class StringContainer:

    def __init__(self):
        self.string = ''

    def append(self, sth: str):
        self.string = self.string + sth


class Node:

    def __init__(self, letter: chr):
        self._letter = letter
        self._neighbours: Dict[chr, Node] = {}
        self._letters_count: Dict[chr, int] = {}

    def add_son(self, letter: chr) -> Node:
        if letter not in self._neighbours:
            self._neighbours[letter] = Node(letter)
            self._letters_count[letter] = 0

        self._letters_count[letter] += 1
        return self._neighbours[letter]

    def get_son(self, letter: chr) -> Node:
        if letter in self._neighbours:
            return self._neighbours[letter]

    def get_sons(self) -> List[Node]:
        return list(self._neighbours.values())

    def get_letter_count(self, letter: chr) -> int:
        count = 0
        if letter in self._letters_count:
            count = self._letters_count[letter]
        return count

    def get_letters_count(self) -> Dict[chr, int]:
        return self._letters_count


class MarkovModel:

    def order(self) -> int:
        raise NotImplementedError()

    def to_string(self) -> str:
        raise NotImplementedError()

    def k_freq(self, k_gram: str) -> float:
        raise NotImplementedError()

    def k_follow_freq(self, k_gram: str, letter: chr) -> float:
        raise NotImplementedError()

    def next_char(self, k_gram: str) -> chr:
        raise NotImplementedError()


class MarkovModelTables(MarkovModel):

    def __init__(self, text: str, k: int):
        self._k = k
        self._k_gram_count = {}
        self._k_gram_letter = {}
        text_len = len(text)

        for left_idx in range(text_len):
            right_idx = left_idx + k
            if right_idx < text_len:
                k_gram = text[left_idx:right_idx]
            else:
                k_gram = text[left_idx:text_len] + text[:right_idx % text_len]

            if k_gram not in self._k_gram_count:
                self._k_gram_count[k_gram] = 0

            if k_gram not in self._k_gram_letter:
                # work with ASCII symbols
                self._k_gram_letter[k_gram] = [0] * 128

            self._k_gram_count[k_gram] += 1
            self._k_gram_letter[k_gram][ord(text[right_idx % text_len])] += 1

    def order(self) -> int:
        return self._k

    def to_string(self) -> str:
        return "\n".join(
            [f"{k}: {[f'{chr(i)} {v[i]}' for i in range(128) if v[i] != 0]}" for k, v in self._k_gram_letter.items()])

    def k_freq(self, k_gram: str) -> float:
        return self._k_gram_count[k_gram]

    def k_follow_freq(self, k_gram: str, letter: chr) -> float:
        return self._k_gram_letter[k_gram][ord(letter)]

    def next_char(self, k_gram: str) -> chr:
        idx = stdrandom.discrete(self._k_gram_letter[k_gram])
        return chr(idx)


class MarkovModelTree(MarkovModel):

    def __init__(self, text: str, k: int):
        self._k = k
        self._root = Node(None)
        k = k+1

        for left_idx in range(len(text)):
            right_idx = left_idx + k
            if right_idx < len(text):
                k_gram = text[left_idx:right_idx]
            else:
                k_gram = text[left_idx:len(text)] + text[:right_idx % len(text)]

            node: Node = self._root

            for ch in k_gram:
                node = node.add_son(ch)

    def order(self) -> int:
        return self._k

    def to_string(self) -> str:
        result = StringContainer()
        MarkovModelTree._stringify(self._k, self._root, '', result)
        return "\n".join(sorted(result.string.split("\n")))

    @staticmethod
    def _stringify(level: int, node: Node, path: str, result: StringContainer):
        if level == 0:
            result.append(f"{path}: {' '.join([f'{k} {v}' for k, v in node.get_letters_count().items()])}\n")
        else:
            [MarkovModelTree._stringify(level - 1, n, path + n._letter, result) for n in node.get_sons()]

    def k_freq(self, k_gram: str) -> float:
        node: Node = self._root
        for i in range(len(k_gram) - 1):
            node = node.get_son(k_gram[i])
        k_gram_count = node.get_letter_count(k_gram[-1])

        return k_gram_count

    def k_follow_freq(self, k_gram: str, letter: chr) -> float:
        node: Node = self._root
        for ch in k_gram:
            node = node.get_son(ch)
        return node.get_letter_count(letter)

    def next_char(self, k_gram: str) -> chr:
        node: Node = self._root
        for ch in k_gram:
            node = node.get_son(ch)
        idx = stdrandom.discrete(list(node.get_letters_count().values()))
        return list(node.get_letters_count().keys())[idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates a path for Markov Model of order k and length t')
    parser.add_argument('order', metavar='k', type=int, help='model order')
    parser.add_argument('trajectory', metavar='t', type=int, help='trajectory length')
    parser.add_argument('file', metavar='f', type=str, help='training file')

    args = parser.parse_args()

    with open(args.file, 'r') as f:
        txt = "\n".join([x.strip().strip("\n") for x in f.readlines()])

    mm = MarkovModelTree(txt, args.order)
    mmt = MarkovModelTables(txt, args.order)

    state = txt[:args.order]

    tj = state
    for _ in range(args.trajectory - args.order):
        c = mm.next_char(state)
        state = state[1:] + c
        tj += c

    print(tj)
