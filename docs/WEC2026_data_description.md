# WEC2026_data_description

Source PDF: [WEC2026_data_description.pdf](./WEC2026_data_description.pdf)

## Page 1

1

WEC 2026 data description
Main dataset for the contest

players_quarters_final.csv
(3 486 rows) – player activity snapshots at match quarter -hour
checkpoints
This is the main analytical dataset for the contest.
It was constructed by aggregating event -level data from selected source tables  (see
below) – player_appearance_shot_limited.csv and player_appearance_run.csv into a
single player-level panel organised around fixed time checkpoints within each match.
The dataset covers 31 matches and 869 unique player appearances. Each row represents
one player at one time checkpoint, capturing their activity both in the preceding 15-minute
window and cumulatively from th e start of the match up to that point. The binary target
variable scored_after indicates whether the player scored a goal at any point after the
checkpoint in the same match, excluding own goals.
Observation unit and time structure.  Checkpoints are defined within match periods,
with minutes counted locally from the start of each period. Up to seven checkpoints are
possible per match: minute 15, 30, and 45 of the first half ( H1_15, H1_30, H1_45), minute
15 and 30 of the second half (H2_15, H2_30), and – for matches that reached extra time –
minute 45 of the second half (H2_45) and minute 15 of the first extra time period (ET1_15).
Only players who were actively on the pitch at the moment of the checkpoint are included,
identified through the continuous minute_in and minute_out fields.
Feature groups. Variables with the prefix last15_ describe the player’s activity in the 15-
minute window immediately preceding the checkpoint; variables with the prefix cumul_
describe their accumulated ac tivity from the start of the match. Both groups cover the
same two domains: physical exertion (derived from player_appearance_run.csv) and
shooting (derived from player_appearance_shot_limited.csv). Goals scored prior to the
checkpoint were excluded from the shot features to prevent information leakage into the
target variable.
Target variable. scored_after is a binary indicator equal to 1 if the player scored at least
one non-own goal after the checkpoint in that match, and 0 otherwise. The target is heavily
imbalanced: 203 positive cases out of 3,486 observations (approximately 5.8%), reflecting
the natural rarity of goalscoring events in football.

## Page 2

2

Table 1. Description of columns in the main dataset
players_quarters_final.csv
Column Description
player_appearance_id Unique player-appearance identifier
player_id Player identifier
fixture_id Match identifier
date Match date (YYYY-MM-DD)
checkpoint Time checkpoint label: H1_15, H1_30,
H1_45, H2_15, H2_30, H2_45, ET1_15
checkpoint_period Match period of the checkpoint: half_1,
half_2, extra_time_1
checkpoint_min Local minute within the period at which the
checkpoint is defined (15, 30, or 45)
position Player position: G (goalkeeper), D
(defender), M (midfielder), A (attacker)
is_home Whether the player’s team played at home
(TRUE/FALSE)
formation Tactical formation of the player’s team
(e.g. 4-1-4-1, 3-4-3)
minute_in Minute the player entered the pitch,
counted continuously from 1 across the full
match
minute_out Minute the player left the pitch, counted
continuously from 1 across the full match
subbed Whether the player was substituted on or
off (TRUE/FALSE)
jersey_number Shirt number
last15_sprints Number of sprints (maximal-speed runs) in
the last 15 minutes before the checkpoint
last15_hsr Number of high-speed runs (sub-maximal)
in the last 15 minutes before the
checkpoint
last15_distance Total distance covered in high -intensity
runs (m) in the last 15 minutes
last15_mean_max_speed Mean of per-run maximum speeds (m/s) in
the last 15 minutes
last15_peak_speed Highest maximum speed recorded across
all runs (m/s) in the last 15 minutes
last15_shots Number of shots (excluding own goals) in
the last 15 minutes

## Page 3

3

Column Description
last15_shots_on_target Number of shots on target (goal or saved)
in the last 15 minutes
last15_shots_under_press Number of shots taken while under
opponent pressure in the last 15 minutes
last15_shots_top_third Number of shots originating from the
attacking third in the last 15 minutes
cumul_sprints Cumulative number of sprints from match
start to the checkpoint
cumul_hsr Cumulative number of high -speed runs
from match start to the checkpoint
cumul_distance Cumulative distance covered in high -
intensity runs (m) from match start to the
checkpoint
cumul_mean_max_speed Mean of per -run maximum speeds (m/s)
from match start to the checkpoint
cumul_peak_speed Highest maximum speed recorded across
all runs (m/s) from match start to the
checkpoint
cumul_shots Cumulative number of shots (excluding
own go als) from match start to the
checkpoint
cumul_shots_on_target Cumulative number of shots on target from
match start to the checkpoint
cumul_shots_under_press Cumulative number of shots taken under
pressure from match start to the
checkpoint
cumul_shots_top_third Cumulative number of shots from the
attacking third from match start to the
checkpoint
scored_after Target variable: 1 if the player scored at
least one goal after the checkpoint in this
match, 0 otherwise (own goals excluded)

## Page 4

4

Source and supplementary datasets
for potential extension of feature space

player_appearance_pass.csv
(29 795 rows) – individual pass events
This table is the most granular passing log in the dataset, with one row per pass attempt
made by a player during a m atch. Across nearly 30,000 rows, it captures every time a
player chose to play the ball to a teammate — or tried to.
Each record is anchored to a specific player appearance via player_appearance_id,
meaning passes can always be traced back to who made them , which team they played
for, and in which match. When a pass has an intended recipient, the
addressee_player_appearance_id field links to that receiver’s appearance record — but
this can be null, which likely indicates a clearance, a speculative long ball , or a situation
where no specific target was identifiable.
The accurate flag is the core quality signal: it marks whether the pass actually reached
its intended target. Combined with the total number of rows per player, this allows
straightforward computation of pass volume and pass accuracy at any desired level of
aggregation (per player, per time interval, per pitch zone).
The table provides two key contextual dimensions. period tells us when in the match the
pass occurred — distinguishing the first and second halves as well as both extra time
periods — while stage encodes where on the pitch  the pass originated, using a three -
zone model: bottom (defensive third), middle (central area), and top (attacking third). The
minute field adds finer temporal resolution within each period. Together, these allow
analysis of how a player’s passing behaviour evolves across the match and shifts across
different areas of the pitch.
This table was NOT used to create features i ncluded in the main dataset for the
contest.

## Page 5

5

Table 2. Description of columns in the supplementary dataset
player_appearance_pass.csv
Column Description
id Unique pass event identifier (integer PK)
period Match period: half_1, half_2,
extra_time_1, extra_time_2
player_appearance_id FK to player_appearance.id – the passer
addressee_player_appearance_id FK to player_appearance.id – the
intended receiver (NULL if no target)
accurate Whether the pass reached its target
(True/False)
minute Minute of t he match when the pass
occurred
stage Pitch zone where the pass originated:
bottom, middle, top, or empty string

Figure 1. Illustration of pitch zones (stages)

bottom (defensive third)                middle (central area)               top (attacking third)

## Page 6

6

player_appearance_behaviour_under_pressure.csv
(12 185 rows) – passes made while being pressed
Where the pass table captures all passing events, this table zooms in on a specific and
tactically rich subset: moments when a player receives and attempts to play the ball while
being actively pressed by an opponent. With over 12,000 rows, it reflects how frequently
pressing occurs in these matches and how players respond when under duress.
The structure shares several fields with the pass table — period, minute, stage, accurate,
and the passer/receiver appearance IDs — but adds three fields that make this table
uniquely informative. First, pressing_player_appearance_id identifies who applied the
pressure, enabling analysis from the pressing player’s perspective as well as the pressed
player’s. Second, press_induced_outcome captures what actually happened as a result of
the press: the player may have played a forward_pass, backward_pass, or lateral_pass,
carried the ball (ball_carry), or lost possession entirely (turnover). This makes it possible
to measure not just whether a player was accurate under pressure, but whether the press
succeeded in disrupting the direction of play. Third, pass_angle records the angle of the
resulting pass in degrees — a continuous variable that is null when the outcome was a
turnover or ball carry (i.e. no pass was made). This opens the door to geometric analysis
of how pressure distorts passing direction.
Together, these fields make this one of the richest tables in the dataset for tactical
analysis. It can answer questions like: which players consistently play forward under
pressure? Which positions are pressed most of ten? Does pressing in the attacking third
yield more turnovers than pressing in the defensive third?
This table was NOT used to create features included in the main dataset for the
contest.

## Page 7

7

Table 3. Description of columns in the supplementary dataset
player_appearance_behaviour_under_pressure.csv
Column Description
id Unique event identifier (integer PK)
period Match period: half_1, half_2,
extra_time_1, extra_time_2
player_appearance_id FK to player_appearance.id – the player
under pressure
addressee_player_appearance_id FK to player_appearance.id – intended
receiver (NULL if turnover/no target)
accurate Whether the pass was accurate
(True/False)
pressing_player_appearance_id FK to player_appearance.id – the player
applying pressure
press_induced_outcome Outcome of the press: forward_pass,
backward_pass, lateral_pass,
ball_carry, turnover
pass_angle Angle of the resulting pass in degrees
(NULL when not applicable, e.g. turnover)
minute Minute of the match
stage Pitch zone: bottom, middle, top, or empty
string

## Page 8

8

player_appearance_run.csv
(35 133 rows) – high-speed run / sprint events
This table captures the physical exertion dimension of each player’s match, logging every
high-intensity running effort they make. With over 35,000 rows — the largest event table
in the dataset — it reflects just how frequently players make explosive movements during
a match.
Each row represents a single running bout, classified by run_type into two categories: hsr
(high-speed running, intense but sub -maximal effort) and sprint (maximal or near -
maximal speed). This distinction is standard in sports science and reflects meaningfully
different physical demands on the body. The table records both the min_speed and
max_speed reached during each run in metres per se cond, as well as the total distance
covered — allowing computation of peak output, average intensity, and cumulative
physical load per player per match.
Contextual fields mirror those in the other event tables: period and minute place each run
in time, while stage places it on the pitch. There is also a possession identifier, which links
runs to specific possession sequences — a detail that enables more advanced analysis,
such as whether players make more high -speed runs during t heir team’s own attacks
versus defensive transitions. This makes the table valuable not just for physical load
analysis but also for understanding the tactical contexts in which players make their most
intense efforts.
This table was used to create some features included in the main dataset for the
contest, but they can be potentially further extended.

## Page 9

9

Table 4. Description of columns in the supplementary dataset
player_appearance_run.csv
Column Description
id Unique run event identifier (integer PK)
period Match period: half_1, half_2,
extra_time_1, extra_time_2
stage Pitch zone: bottom, middle, top
possession Possession sequence identifier (integer)
run_type Type of run: hsr (high-speed run) or
sprint
minute Minute of the match
min_speed Minimum speed during the run (m/s,
decimal)
max_speed Maximum speed during the run (m/s,
decimal)
distance Distance covered during the run (meters,
decimal)
player_appearance_id FK to player_appearance.id

## Page 10

10

player_appearance_shot_limited.csv
(780 rows) – individual shot events
This table logs every shot attempt across all matches in the dataset, with one row per shot
event. At 780 rows it is considerably smaller than the passing and running tables –
reflecting the natural scarcity of shooting moments in football – but it is arguably the most
information-dense table in the dataset, packing a rich set of contextual descriptors around
each attempt.
Each shot is linked to the shooter via player_appearance_id, placing it within a specific
player’s appearance in a specific match. The period and minute fields situate the attempt
in time, while stage places it on the pitch using the same three -zone model ( bottom,
middle, top) used throughout the dataset. The possession identifier connects each shot
to a broader atta cking sequence, which makes it possible to study how moves develop
before they culminate in an attempt.
Where this table really stands out is in the detail it captures about how each shot was
taken and why it ended the way it did. body_part records whether the attempt was made
with the right foot, left foot, head, or another body part – a basic but important dimension
of shot quality. technique goes further, distinguishing normal strikes from volleys, lobs,
and overhead kicks, each of which carries very dif ferent expectations of accuracy and
power. play_pattern describes how the attacking move was constructed: whether it came
from regular build -up play, a counter -attack, a set piece (corner, direct or indirect free
kick), a penalty, or even a throw -in – giving a full picture of the tactical context in which
the shot arose.
The outcome field is the central variable in the table, recording what actually happened:
the shot may have resulted in a goal, been on_target (saved), gone off_target, been
blocked, or str uck the hit_woodwork. Two additional outcomes – own_goal and
disallowed_goal – cover edge cases, and the table handles these carefully through
dedicated foreign keys. own_goal_player_appearance_id identifies the defending player
who inadvertently scored, a nd is null for all non -own-goal events. Similarly,
block_player_appearance_id identifies the player who blocked the shot, and is null
whenever the shot was not blocked. This design means both the shooter’s and the
blocker’s or own-goal scorer’s perspectives can be recovered from a single row. Note that
outcome is excluded from the version of this table provided for the contest. Because
the target variable scored_after in the main dataset is derived directly from whether
a shot resulted in a goal, including outcome as a feature would constitute information
leakage – a model trained on it would effectively be given the answer rather than
learning to predict it.
Finally, the under_pressure flag ties this table back to the pressing theme that runs
through the da taset: it marks whether the shooter was being actively pressed at the
moment of the attempt, enabling direct comparison of shot quality and outcome rates
between pressured and unpressured situations.
This table was used to create some features included in the main dataset for the
contest, but they can be potentially further extended.

## Page 11

11

Table 5. Description of columns in the supplementary dataset
player_appearance_shot_limited.csv
Column Description
id Unique shot event identifier (integer PK)
period Match period: half_1, half_2,
extra_time_1, extra_time_2
player_appearance_id FK to player_appearance.id – the shooter
body_part Body part used: right_foot, left_foot,
head, other
technique Shot technique: normal, volley, lob,
overhead_kick, other
play_pattern How the attack was built: regular_play,
counter_attack, corner_kick,
direct_free_kick, indirect_free_kick,
penalty, throw_in
own_goal_player_appearance_id FK to player_appearance.id – the player
who scored an own goal (NULL otherwise)
block_player_appearance_id FK to player_appearance.id – the player
who blocked the shot (NULL if not blocked)
minute Minute of the match
possession Possession sequence identifier (integer)
stage Pitch zone: bottom, middle, top
under_pressure Whether the shooter was under pressure
(True/False)
