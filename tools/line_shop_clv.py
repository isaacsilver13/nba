"""Model-free line-shopping / stale-book test on the SGO captures (no prediction model, so no model leak to hide).

    python tools/line_shop_clv.py                 # reads $NBA_DATA_DIR/raw/odds/sgo/**/*.parquet (git pull the data repo first)
    python tools/line_shop_clv.py --selftest

Per game (full-game moneyline: period_id "game", bet_type_id "ml", stat_id "points"; FanDuel's firstToScore rows are also
typed "ml" and are excluded): every book's ENTRY and CLOSE American odds, de-vigged per book (multiplicative). In the code
and the output the entry price is still called "open" (open_h, p_open, loo_open), so read "open" below as "entry".
  entry="snapshot" (default): each book's price in the same ~10am capture (slot open/props), so all books are priced at one
      moment and a bet could actually have been placed. Close = SGO's closeOdds from the settle capture.
  entry="opening" (--entry opening): each book's own openOdds. Books post these at different times, so comparing one
      book's open with the others' opens looks ahead (posting-time look-ahead); kept only to show that artifact.
  fair_open_loo(b)  = median over the OTHER books' de-vigged entry prob   (leave-one-out so a book never votes on itself)
  fair_close_loo(b) = median over the OTHER books' de-vigged close prob
Strategies priced at a book's entry, selected using only entry-time information, scored by CLV against the others' close:
  soft-book : bet side s at book b when dec_b(s) * fair_open_loo(b)[s] - 1 > THR   (is any book's price stale vs the rest?)
  shop      : bet the best entry decimal across books for the side with the biggest edge vs the leave-one-out consensus
  baseline  : every side at every book (= minus the vig; the number everything must beat)
CLV = fair_close_loo[s] * dec_entry - 1, i.e. the expected return IF the others' close is efficient; it already pays the vig.
A data-integrity filter drops books whose close sits persistently far from the others' (MAX_BOOK_DEV): that is mis-mapped
data, not a soft book.

Caveats printed with every run: limits at these prices may be tiny and a price may be gone within minutes, so a positive
CLV here is an upper bound on what is bettable; n is small early in a season (SE is printed; < 300 games is descriptive only).
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NBA_DATA_DIR", ROOT / "data"))
THR = 0.02
MIN_BOOKS = 4          # need >= 3 other books for a leave-one-out median
MAX_BOOK_DEV = 0.03    # mean |de-vigged close prob - others' median| above this = data problem, not a soft book


def dec(american):
    a = np.asarray(american, float)
    return np.where(a > 0, 1 + a / 100, 1 + 100 / -a)


def load(path, entry="snapshot"):
    """One row per (event, book): home/away entry & close decimal.

    entry="snapshot" (default): every book's price in the same ~10am capture (slot open/props), so all books are priced at
      one moment and a bet could actually have been placed. Close = SGO's closeOdds from the settle capture.
    entry="opening": each book's own openOdds. Books post these at DIFFERENT times (it equals the 10am price only ~16% of
      the time), so comparing a book's open with the others' opens looks ahead; kept only to show that artifact.
    """
    d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(Path(path) / "*" / "*.parquet")))], ignore_index=True)
    # stat_id == "points" is the real moneyline; "firstToScore" etc. also come as bet_type "ml" (FanDuel posts them) and are not it
    d = d[(d.period_id == "game") & (d.bet_type_id == "ml") & (d.stat_id == "points")].sort_values("captured_at_utc")
    k = ["event_id", "bookmaker_id", "side_id"]
    close = d[d.close_odds.notna()].groupby(k).close_odds.last()
    if entry == "snapshot":
        px = d[d.slot.str.split("@").str[0].isin(["open", "props"]) & d.odds_american.notna()].groupby(k).odds_american.last()
    else:
        px = d[d.open_odds.notna()].groupby(k).open_odds.last()
    d = pd.concat({"open_odds": px, "close_odds": close}, axis=1).dropna().reset_index()
    if d.empty:
        return pd.DataFrame()
    w = d.pivot(index=["event_id", "bookmaker_id"], columns="side_id", values=["open_odds", "close_odds"])
    w = w.dropna().reset_index()
    out = pd.DataFrame({"event_id": w.event_id, "book": w.bookmaker_id})
    for t in ("open", "close"):
        out[f"{t}_h"], out[f"{t}_a"] = dec(w[(f"{t}_odds", "home")]), dec(w[(f"{t}_odds", "away")])
    return out


def prepare(w):
    """Add de-vigged home prob per book and leave-one-out medians of it; drop games with too few books."""
    w = w.copy()
    for t in ("open", "close"):
        ih, ia = 1 / w[f"{t}_h"], 1 / w[f"{t}_a"]
        w[f"p_{t}"] = ih / (ih + ia)
    def loo(w):
        w = w[w.groupby("event_id").book.transform("size") >= MIN_BOOKS].copy()
        for t in ("open", "close"):
            w[f"loo_{t}"] = [w[(w.event_id == e) & (w.book != b)][f"p_{t}"].median() for e, b in zip(w.event_id, w.book)]
        return w
    w = loo(w)
    # integrity: a real stale price converges by the close; a book that sits far from everyone at BOTH times is mis-mapped data
    dev = (w.p_close - w.loo_close).abs().groupby(w.book).mean()
    bad = dev[dev > MAX_BOOK_DEV].index.tolist()
    if bad:
        print(f"  DROPPED books that persistently disagree with the rest (mean |close - others| > {MAX_BOOK_DEV}): "
              + ", ".join(f"{b} ({dev[b]:.3f})" for b in bad))
        w = loo(w[~w.book.isin(bad)])
    return w


def bets(w, thr=THR):
    """Long table of side-bets at each book's open, with CLV vs the others' close."""
    rows = []
    for side, dcol, sgn in (("home", "open_h", 1), ("away", "open_a", -1)):
        p_open = w.loo_open if sgn == 1 else 1 - w.loo_open
        p_close = w.loo_close if sgn == 1 else 1 - w.loo_close
        rows.append(pd.DataFrame({"event_id": w.event_id, "book": w.book, "side": side, "dec": w[dcol],
                                  "edge": w[dcol] * p_open - 1, "clv": w[dcol] * p_close - 1}))
    return pd.concat(rows, ignore_index=True)


def summarize(b, label):
    n = len(b)
    se = b.clv.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    print(f"  {label:34s} n={n:>5} games={b.event_id.nunique():>4}  mean CLV {b.clv.mean():+.4f} (se {se:.4f})")


def report(w):
    b = bets(w)
    print(f"games={w.event_id.nunique()}  books={w.book.nunique()}  book-games={len(w)}")
    summarize(b, "baseline: every side at every book")
    summarize(b[b.edge > THR], f"soft-book: edge vs LOO open > {THR:.0%}")
    top = b.loc[b.groupby(["event_id", "side"]).dec.idxmax()]
    summarize(top, "shop: best open price, both sides")
    sel = top.loc[top.groupby("event_id").edge.idxmax()]
    summarize(sel[sel.edge > 0], "shop: best side only if edge > 0")
    print("  per book, soft-book rule:")
    for k, g in b[b.edge > THR].groupby("book"):
        summarize(g, f"    {k}")
    if b.event_id.nunique() < 300:
        print("  UNDERPOWERED: < 300 games; treat as descriptive only.")
    print("  NOTE: open prices have tiny limits and may not be bettable at all; CLV here is an upper bound.")


def selftest():
    rng = np.random.default_rng(0)
    rows = []
    for e in range(600):
        lt = rng.normal(0, 0.8)                                    # true home logit
        for k, stale in enumerate((0, 0, 0, 0, 0, 1)):             # book 5 is a stale-open book
            def prob(noise):
                return 1 / (1 + np.exp(-(lt + noise)))
            po = prob(rng.normal(0, 0.05) + (rng.normal(0, 0.4) if stale else 0))
            pc = prob(rng.normal(0, 0.05))
            rows.append(dict(event_id=e, book=f"b{k}", open_h=1 / (po * 1.04), open_a=1 / ((1 - po) * 1.04),
                             close_h=1 / (pc * 1.04), close_a=1 / ((1 - pc) * 1.04)))
    b = bets(prepare(pd.DataFrame(rows)))
    base, soft = b.clv.mean(), b[(b.edge > THR) & (b.book == "b5")].clv.mean()
    assert base < -0.02, base                                      # no stale book anywhere = lose the vig
    assert soft > base + 0.03, (soft, base)                        # the stale book is found
    assert soft > b[(b.edge > THR) & (b.book != "b5")].clv.mean() + 0.02   # and it stands out from the sharp books
    print("selftest ok", round(base, 4), round(soft, 4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--path", default=str(DATA / "raw" / "odds" / "sgo"))
    ap.add_argument("--entry", choices=("snapshot", "opening"), default="snapshot", help="see load(); 'opening' looks ahead across books")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        w = load(a.path, a.entry)
        if w.empty:
            raise SystemExit("no rows with both open_odds and close_odds yet (close arrives with the next-morning settle capture); git pull the data repo")
        report(prepare(w))
