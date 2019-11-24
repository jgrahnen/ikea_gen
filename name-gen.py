import pandas as pd
import numpy as np
from collections import defaultdict


class MarkovWordGenerator:
    START_TOKEN = '^'
    STOP_TOKEN = '$'

    def __init__(self, state_size=1):
        # The Markov chain can have states which are several characters long;
        # e.g. 'a', 'an', 'and', etc.
        self.state_size = state_size

        # From each state you have some probability of observing (or when sampling,
        # emitting) the next character
        self.emission_prob = {}

        # We will also keep track of the probability observing any character
        # overall; this will come in handy when we want to sample from a
        # state we never encountered
        self.char_prob = {}

    def fit(self, word_list):
        # We want two magical tokens representing start and end, so we
        # can tell where words start and when to stop sampling
        bracketed_words = ["".join([self.START_TOKEN, word, self.STOP_TOKEN]) for word in word_list]

        # Find the frequency of emitted characters after each state
        for word in bracketed_words:

            # Convert everything to lower-case since we'll get screwed-up counts
            # otherwise
            lc_word = word.lower()

            # Left-pad the word to support the desired state size
            padded_word = " "*(self.state_size-1) + lc_word

            # Observe all the states in the word, and which character is
            # emitted after each state
            for ix in range(len(padded_word)-self.state_size):
                state = padded_word[ix:ix+self.state_size]
                emitted_char = padded_word[ix+self.state_size]

                if state not in self.emission_prob:
                    self.emission_prob[state] = defaultdict(float)
                if emitted_char not in self.char_prob:
                    self.char_prob[emitted_char] = 0.0

                # We're actually looking at emission frequencies here, but we'll
                # convert them to probabilities just below
                self.emission_prob[state][emitted_char] += 1.0
                self.char_prob[emitted_char] += 1.0

        # For ease of later sampling, convert emission frequencies to probabilities
        for state in self.emission_prob:
            emission_freqs = self.emission_prob[state]
            tot_counts = sum(emission_freqs.values())
            emission_probs = np.array(list(emission_freqs.values())) / tot_counts
            self.emission_prob[state] = dict(zip(emission_freqs.keys(), emission_probs))

        tot_char_counts = sum(self.char_prob.values())
        char_probs = np.array(list(self.char_prob.values())) / tot_char_counts
        self.char_prob = dict(zip(self.char_prob.keys(), char_probs))

        return self

    def predict(self, seed_string=None, max_len=None):
        if seed_string is None:
            start_state = " "*(self.state_size-1) + self.START_TOKEN
        else:
            raise NotImplementedError("Haven't implemented custom starting strings yet.")

        if max_len is not None:
            raise NotImplementedError("Haven't implemented maximum sampling length yet")

        # We'll sample new characters in the name based on the emission/observation
        # probabilities we calculated during the fitting procedure
        emitted_token = ""
        sampled_word = ""
        current_state = start_state
        while emitted_token != self.STOP_TOKEN:

            # If we've seen this state (i.e. string of self.state_size characters) before,
            # sample a character that's like to come after it.
            if current_state in self.emission_prob:
                possible_chars = np.array(list(self.emission_prob[current_state].keys()))
                char_probs = np.array(list(self.emission_prob[current_state].values()))
                emitted_token = np.random.choice(possible_chars, 1, p=char_probs)[0]

            # Apparently we've never seen this state (can happen with high state sizes),
            # so just sample a common character to try to get back to a state we have
            # seen.
            else:
                possible_chars = np.array(list(self.char_prob.keys()))
                char_probs = np.array(list(self.char_prob.values()))
                emitted_token = np.random.choice(possible_chars, 1, p=char_probs)[0]

            sampled_word += emitted_token
            current_state = current_state + emitted_token
            current_state = current_state[-self.state_size:]

        # Clean it up so it looks like a name
        sampled_word = sampled_word.lstrip(self.START_TOKEN).rstrip(self.STOP_TOKEN).capitalize()

        return sampled_word


def load_names(file_path, gender='M'):
    # We're expecting name counts on the format provided by
    # the Social Security Administration, e.g. from
    # https://www.ssa.gov/oact/babynames/limits.html
    name_counts = pd.read_csv(file_path, header=0, names=['Name', 'Gender', 'Count'])
    gendered_name_list = name_counts[name_counts.Gender == gender].Name.tolist()

    return gendered_name_list


if __name__ == "__main__":
    baby_names = load_names('yob2018.txt')
    print('Top 5 male names 2018: {}'.format(", ".join(baby_names[:5])))

    name_generator = MarkovWordGenerator(state_size=4).fit(baby_names)
    print('Random new name: {}'.format(name_generator.predict()))
