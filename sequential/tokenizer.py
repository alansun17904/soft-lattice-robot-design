class Vocabulary:
    def __init__(self, robot_size):
        self.word2idx = {
            # "<pad>": 0,
            # "<sos>": 1,
            # "<eos>": 2,
            # "<unk>": 3,
            "<add>": 0,
            "<remove>": 1,
            "<noop>": 2,
        }
        self.idx2word = {
            # 0: "<pad>",
            # 1: "<sos>",
            # 2: "<eos>",
            # 3: "<unk>",
            0: "<add>",
            1: "<remove>",
            2: "<noop>",
        }
        self.idx = 0
        # add all robot coordinates as tokens
        for i in range(robot_size):
            self.word2idx[i] = len(self.word2idx)
            self.idx2word[len(self.idx2word)] = i

    def __len__(self):
        return len(self.word2idx)


class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def tokenize(self, trajectories):
        """
        Tokenize the actions in a list of provided trajectories
        """
        tokenized_trajectories = []
        for trajectory in trajectories:
            # tokenized_trajectory = [self.vocab.word2idx["<sos>"]]
            tokenized_trajectory = []
            for i, action in trajectory:
                if action[0] != 2 and action[1] is not None:
                    if action[0] == 0:
                        token = "<add>"
                    else:
                        token = "<remove>"
                    tokenized_trajectory.append(self.vocab.word2idx[token])
                    tokenized_trajectory.append(self.vocab.word2idx[action[1]])
                if action[0] == 2:
                    tokenized_trajectory.append(self.vocab.word2idx["<noop>"])
            # tokenized_trajectory.append(self.vocab.word2idx["<eos>"])
            tokenized_trajectories.append(tokenized_trajectory)

        return tokenized_trajectories

    def detokenize(self, tokenized_trajectory):
        """
        Detokenize a tokenized trajectory
        """
        return [
            [self.vocab.idx2word[token] for token in trajectory]
            for trajectory in tokenized_trajectory
        ]


if __name__ == "__main__":
    import pickle

    valid_trajectories = pickle.load(open("data/valid_trajectories.pkl", "rb"))
    t = Tokenizer(Vocabulary(9))
    tokenized = t.tokenize(valid_trajectories[:20])
    print(t.vocab.word2idx)
    print(valid_trajectories[0])
    print(tokenized[0])

    # detokenize
    print(t.detokenize(tokenized[:1]))
