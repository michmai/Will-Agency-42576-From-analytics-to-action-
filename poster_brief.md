# Franchise trajectories — poster brief

## The red thread

We were given `Movie_50k.csv` (IMDb-style: titles, ratings, votes, genres). During exploratory work we kept circling back to one question: **what happens to a franchise as it grows — does each sequel get worse, or just different?**

The 50k dataset can't answer that. It has no `belongs_to_collection`, no budget, no revenue. So we pivoted to the **TMDB metadata dump** (~45k movies, ~3k of which form franchises with ≥3 entries) and used that as the basis for the analysis.

From there the narrative is one continuous question: **does it pay to keep making sequels?** We answer it from four angles — rating trajectory, revenue trajectory, genre, and what predicts the *next* movie's quality.

## What we found

1. **Rating holds, then drifts.** Average TMDB rating across franchises stays roughly flat through movies #1–#4, then starts widening (some franchises rebound, most decay). [`outputs/franchise_rating_by_seq.png`]
2. **Revenue is a power law, not a trend.** Median sequel revenue doesn't grow — but a small set of franchises grow enormously, dragging the mean. [`outputs/franchise_revenue_by_seq.png`]
3. **Genre changes the story.** Action franchises hold up best across sequels; some genres decay much faster. [`outputs/franchise_rating_by_seq_by_genre.png`]
4. **The best predictor of the next movie isn't money — it's the franchise's existing rating profile** (average rating, recent trend, last rating). Budget and revenue rank surprisingly low. [`outputs/feature_importance.png`]
5. **Sequel #N drifts further from #1 the higher N gets** — boxplot widens, median stays near zero. The risk grows, the average payoff doesn't. [`outputs/sequel_decay.png`]
6. **Time between sequels barely matters** for rating change — waiting longer doesn't reliably rescue a franchise. [`outputs/time_gap_vs_rating.png`]
7. **Cash cows ≠ prestige.** Top revenue franchises cluster across a wide rating range; high revenue does not require high rating. [`outputs/top_franchises_revenue_vs_rating.png`]

**Bottom line for the poster:** *Sequels are a stability bet, not a growth bet. Most franchises don't get better — but they don't have to, because revenue compounds even when ratings don't.*

---

## Prompt for Figma / poster generator

Paste the block below into Figma AI, FigJam AI, or any layout tool that takes a brief. Replace the bracketed image references with the actual files from `outputs/`.

```
Design a single-page exploratory data visualization poster (A1 portrait, 594×841mm).

Topic: "Do movie franchises get worse? — A trajectory analysis using TMDB metadata"

Tone: editorial, data-journalism style. Think The Pudding / FiveThirtyEight / The Economist Graphic Detail. Clean, generous whitespace, serif headline + sans-serif body. Restrained palette: off-white background, charcoal text, one accent color (deep crimson) for emphasis, secondary blue for revenue.

Narrative structure (top-to-bottom, single red thread):

1. HEADLINE BLOCK (top ~15%)
   - Bold title: "Do franchises get worse with each sequel?"
   - Deck (one sentence): "We started with a 50k-movie IMDb sample, but to follow franchises we switched to TMDB. Here's what 3,000 franchise movies told us."
   - Small byline + dataset note.

2. THE PIVOT (one short paragraph, italic, max 40 words)
   "The dataset we were given couldn't answer this question — no budgets, no collections. So we used TMDB instead."

3. ACT 1 — TRAJECTORY (two charts side by side)
   - Left: [franchise_rating_by_seq.png] — caption: "Rating holds through ~4 movies, then drifts."
   - Right: [franchise_revenue_by_seq.png] — caption: "Revenue is a power law, not a trend."

4. ACT 2 — IT DEPENDS ON GENRE (full-width chart)
   - [franchise_rating_by_seq_by_genre.png] — caption: "Action franchises hold up best; others decay faster."

5. ACT 3 — WHAT ACTUALLY PREDICTS THE NEXT MOVIE (chart + callout)
   - [feature_importance.png] — caption: "Money doesn't predict the next rating. The franchise's existing rating profile does."
   - Pull-quote in the accent color: "Budget and revenue rank surprisingly low."

6. ACT 4 — THE RISK STORY (two charts side by side)
   - Left: [sequel_decay.png] — caption: "The further into a franchise, the wider the spread."
   - Right: [time_gap_vs_rating.png] — caption: "Waiting longer doesn't rescue a franchise."

7. CLOSER (full-width chart)
   - [top_franchises_revenue_vs_rating.png] — caption: "Cash cows aren't prestige. The biggest franchises sit across the entire rating range."

8. TAKEAWAY BLOCK (bottom, boxed, accent color)
   "Sequels are a stability bet, not a growth bet. Most franchises don't get better — but they don't have to, because revenue compounds even when ratings don't."

Layout rules:
- Each "Act" is visually separated by a thin horizontal rule and a small numeral (1/2/3/4).
- All chart captions use the same sans-serif at the same size.
- Use the accent crimson only for: the title, the takeaway box border, and key data points (e.g., the mean line in chart 1, the zero-line in chart 5).
- No 3D, no gradients, no drop shadows.
- Footer: small dataset note "TMDB metadata, n≈3,000 franchise movies, ≥3 entries per franchise" and a credit line.
```
