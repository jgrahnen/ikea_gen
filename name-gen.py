import pandas as pd


class MarkovWordGenerator:
    def __init__(self, state_size=1):
        self.state_size = state_size

    def fit(self, word_list):
        return self

    def predict(self, seed_string=None, max_len=None):
        return "Johan"


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

    name_generator = MarkovWordGenerator(state_size=3).fit(baby_names)
    print('Random name: {}'.format(name_generator.predict()))
