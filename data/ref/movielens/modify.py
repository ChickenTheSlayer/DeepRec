import os.path as op

def get_movie_id2name(data_dir='.'):
    movie_id2name = {}
    item_path = op.join(data_dir, 'u.item')
    print(f"Loading {item_path}")
    with open(item_path, 'r', encoding="ISO-8859-1") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            ll = line.strip('\n').split('|')
            movie_id = int(ll[0]) - 1  # 0-indexed
            title = ll[1][:-7]  # Remove year
            for sub_s in [", The", ", A", ", An"]:
                if sub_s in title:
                    title = sub_s[2:] + " " + title.replace(sub_s, "")
            movie_id2name[movie_id] = title
            if i < 5 or i >= len(lines) - 5:
                print(f"ID {movie_id}: {title}")
    print(f"Total movies loaded: {len(movie_id2name)}")
    return movie_id2name

data_dir = '/home/msai/weichen001/.virtualenvs/LLaRA/data/ref/movielens'
movie_id2name = get_movie_id2name(data_dir)