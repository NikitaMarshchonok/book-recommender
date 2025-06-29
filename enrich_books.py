import pathlib, pandas as pd

DATA = pathlib.Path("data")

books      = pd.read_csv(DATA / "books.csv")          # book_id
book_tags  = pd.read_csv(DATA / "book_tags.csv")      # goodreads_book_id
tags       = pd.read_csv(DATA / "tags.csv")           # tag_id, tag_name

# --- привести названия к одному формату ---------------------------------
book_tags = book_tags.rename(columns={"goodreads_book_id": "book_id"})

# --- book_id | tag_name | count -----------------------------------------
bt = (book_tags
      .merge(tags, on="tag_id")                       # + tag_name
      .sort_values(["book_id", "count"], ascending=[True, False]))

# --- самый популярный тег для каждой книги ------------------------------
top_tags = bt.groupby("book_id").first()["tag_name"]  # Series

# --- присоединяем к books ------------------------------------------------
books["tag"] = books.book_id.map(top_tags)

books.to_csv(DATA / "books_plus.csv", index=False)

print("✅ books_plus.csv saved  |  books with tag:",
      books.tag.notna().sum(), "/ 10000",
      "|  unique tags:", books.tag.nunique())

