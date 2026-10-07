"""Paper metadata -> Postgres."""
from datetime import date

from sqlalchemy import JSON, Date, Integer, String, Text, create_engine
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from sdrag.config import DATABASE_URL


class Base(DeclarativeBase):
    pass


class Paper(Base):
    __tablename__ = "papers"
    arxiv_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    authors: Mapped[list] = mapped_column(JSON)
    abstract: Mapped[str] = mapped_column(Text)
    published: Mapped[date] = mapped_column(Date)
    categories: Mapped[list] = mapped_column(JSON)
    pdf_url: Mapped[str] = mapped_column(Text)
    n_chars: Mapped[int] = mapped_column(Integer, default=0)
    n_chunks: Mapped[int] = mapped_column(Integer, default=0)


def upsert_papers(papers: list[dict]) -> int:
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    cols = {c.name for c in Paper.__table__.columns}
    rows = [{k: v for k, v in p.items() if k in cols} for p in papers]
    for r in rows:
        r["published"] = date.fromisoformat(r["published"])
    stmt = insert(Paper).values(rows)
    stmt = stmt.on_conflict_do_update(index_elements=["arxiv_id"],
                                      set_={c: stmt.excluded[c] for c in cols - {"arxiv_id"}})
    with engine.begin() as conn:
        conn.execute(stmt)
    return len(rows)
