# Will-Agency-42576-From-analytics-to-action-

## Project idea
**Hidden Gems + Drivers of Quality vs Reach (IMDb)**
- **Quality:** `imdbRating`
- **Reach:** `numberOfVotes` (use `log10`)

## Scope (keep it tight)
- Filter to `titleType == "movie"` and `isAdult == 0`
- Time window: 1990–2025
- Analyze both: (A) all titles and (B) “reliably rated” (`numberOfVotes >= 1000`)

## To-do
1) **Prep**
   - Clean missing/outliers (year/runtime), split multi-valued fields (`genres`, `keywords`, countries/languages)
   - Create: `logVotes`, `decade`, `runtimeBin`, `isEnglish`, `genreCount`, `keywordCount`

2) **EDA (exploratory process)**
   - Distributions: rating, votes (log), runtime, releaseYear
   - Segment comparisons: rating & reach by `genres`, `decade`, `mainCountry`, `firstLanguage`
   - Relationship checks: runtime↔rating, year↔rating, year↔reach
   - Sensitivity: rerun key plots with vote thresholds

3) **Define “Hidden Gems”**
   - High quality (e.g., top 10% rating)
   - Underexposed: low votes **within same decade + genre** (controls time/genre bias)

## Visuals to ship (presentation-ready)
- Rating + log(votes) distributions
- Scatter: `imdbRating` vs `logVotes` (color=genre, size=votes)
- Genre rating boxplots + genre reach comparison
- Trend: median rating by decade + title count by decade
- “Hidden gems” list + highlight in scatter

## Early recommendations (examples)
- Marketing: promote high-rating/low-vote segments (“hidden gems”)
- Content: balance portfolio between high-reach mainstream and high-quality niches
- Product: personalize differently for “quality seekers” vs “mass appeal” viewers

## Optional (only if time)
Light prediction to support the story:
- Model `imdbRating` (regression) or “high rating” (classification) from runtime/year/genres/language + simple counts
- Report drivers; frame as association, not causation

## Reflections on datafication (must cover)
Selection bias (who rates), popularity/time bias (votes accumulate), label noise (genres/keywords), ratings/votes ≠ business value.
