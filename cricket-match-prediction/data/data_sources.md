# Data Sources Documentation

## Primary Dataset: Cricsheet Indian Premier League (IPL) Matches
- **Source**: [Cricsheet.org](https://cricsheet.org/downloads/)
- **Download URL**: `https://cricsheet.org/downloads/ipl_csv2.zip`
- **License**: [Open Data Commons Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/odbl/)
- **Match Format**: Twenty20 (T20) — Indian Premier League
- **Date Range Covered**: 2008/04/18 to 2026/05/31 (Seasons: 2007/08 – 2026)
- **Total Records**: 1243 matches
- **Total Columns**: 16

## Raw Schema (`data/raw/matches.csv`)
| Column | Description |
| :--- | :--- |
| `match_id` | Unique numerical identifier for each match from Cricsheet |
| `season` | IPL tournament edition/year |
| `date` | Date of the match (YYYY/MM/DD) |
| `city` | Host city |
| `venue` | Stadium/ground name |
| `team1` | First team (as recorded in match info) |
| `team2` | Second team |
| `toss_winner` | Team winning the coin toss |
| `toss_decision` | Decision taken upon winning toss (`bat` or `field`) |
| `winner` | Match winner |
| `eliminator` | Winner determined by Super Over / Eliminator (if tie) |
| `result` | Result descriptor (`normal`, `tie`, `no result`) |
| `win_by_runs` | Margin of victory in runs (if batting first) |
| `win_by_wickets` | Margin of victory in wickets (if chasing) |
| `method` | DLS / Rain adjustment method (if applicable) |
| `player_of_match` | Recipient of Player of the Match |
