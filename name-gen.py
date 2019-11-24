import pandas as pd


def load_names(file_path, gender='M'):
    # We're expecting name counts on the format provided by
    # the Social Security Administration, e.g. from
    # https://www.ssa.gov/oact/babynames/limits.html
    name_counts = pd.read_csv(file_path, header=0, names=['Name', 'Gender', 'Count'])
    gendered_name_list = name_counts[name_counts.Gender == gender].Name.tolist()

    return gendered_name_list


if __name__ == "__main__":
    baby_names = load_names('yob2018.txt')
    print(baby_names[:5])