# Running experiments


## MVP

# 1. Set your API key
echo "ANTHROPIC_API_KEY=sk-..." > .env

# 2. Download plays
conda run -n uta_model python scripts/download_plays.py

# 3. Run analysis (builds Lopakhin's bible)
conda run -n uta_model python scripts/run_analysis.py cherry_orchard --characters LOPAKHIN

> Note: Cost of running with Claude Open 4.6 on Cherry Orchard is estimated to be $7.50-10

# 4. Improvise
conda run -n uta_model python scripts/run_improvisation.py session \
  --character LOPAKHIN --play cherry_orchard \
  --setting "A Moscow office, winter" --stakes "Everything he built is at risk"

## Clearing cache

If there is a parsing issue, you may need to clear the beats cache in order to regenerate it:

```bash
rm data/beats/cherry_orchard_beats.json
```