import pandas as pd
import numpy as np

np.random.seed(7)
teams = ['India', 'Australia', 'England', 'South Africa', 'New Zealand',
         'Pakistan', 'Sri Lanka', 'West Indies']
venues = ['Home', 'Away', 'Neutral']

n = 1200
rows = []
for i in range(n):
    team_a = np.random.choice(teams)
    team_b = np.random.choice([t for t in teams if t != team_a])
    venue = np.random.choice(venues, p=[0.4, 0.4, 0.2])
    toss_winner = np.random.choice([team_a, team_b])
    toss_decision = np.random.choice(['bat', 'field'])
    team_a_recent_form = np.random.randint(0, 6)   # wins in last 5 matches
    team_b_recent_form = np.random.randint(0, 6)
    team_a_avg_score = np.random.normal(260, 30)
    team_b_avg_score = np.random.normal(260, 30)
    h2h_a_wins = np.random.randint(0, 10)
    h2h_b_wins = np.random.randint(0, 10)

    # underlying "skill" signal that determines win probability
    skill_diff = (team_a_recent_form - team_b_recent_form) * 3 \
                 + (team_a_avg_score - team_b_avg_score) * 0.15 \
                 + (h2h_a_wins - h2h_b_wins) * 1.5 \
                 + (5 if venue == 'Home' and toss_winner == team_a else 0) \
                 + np.random.normal(0, 12)  # randomness/upsets

    team_a_wins = 1 if skill_diff > 0 else 0

    rows.append([team_a, team_b, venue, toss_winner, toss_decision,
                 team_a_recent_form, team_b_recent_form,
                 round(team_a_avg_score, 1), round(team_b_avg_score, 1),
                 h2h_a_wins, h2h_b_wins, team_a_wins])

df = pd.DataFrame(rows, columns=[
    'TeamA', 'TeamB', 'Venue', 'TossWinner', 'TossDecision',
    'TeamA_RecentForm', 'TeamB_RecentForm', 'TeamA_AvgScore', 'TeamB_AvgScore',
    'H2H_A_Wins', 'H2H_B_Wins', 'TeamA_Wins'
])
df.to_csv('/home/claude/portfolio/cricket-match-prediction/data/matches.csv', index=False)
print(df.shape)
print(df['TeamA_Wins'].value_counts(normalize=True))
