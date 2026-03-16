# Running experiments

# 1. Set your API key
echo "ANTHROPIC_API_KEY=sk-..." > .env

# 2. Download plays

```bash
python scripts/download_plays.py
```

# 3. Run analysis (builds Lopakhin's bible)

```bash
python scripts/run_analysis.py cherry_orchard --characters LOPAKHIN
```

> Note: Cost of running with Claude Open 4.6 on Cherry Orchard was $10.21, and the cost on Hamlet was $15.39.

# 4. Improvise

```bash
python scripts/run_improvisation.py session \
  --character LOPAKHIN --play cherry_orchard \
  --setting "A Moscow office, winter" --stakes "Everything he built is at risk"
```

```bash
python scripts/run_improvisation.py session \
  --character HAMLET --play hamlet \
  --setting "An office kitchenette in Scranton" --stakes "Jeff has taken Hamlet's pizza from the fridge and is pretending its his own"
```


## Clearing cache

If there is a parsing issue, you may need to clear the beats cache in order to regenerate it:

```bash
rm data/beats/cherry_orchard_beats.json
```