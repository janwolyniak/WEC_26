# WEC2026_Problem_description

Source PDF: [WEC2026_Problem_description.pdf](./WEC2026_Problem_description.pdf)

## Page 1

1

WEC 2026 problem description

Will you score a goal?
Predicting goal-scoring probability in football using performance
match data
In modern professional sport and football in particular, real -time and post -match data
analysis has become a standard analytical support tool used by most coaching staffs
worldwide. One of the most important application areas of such analysis is the assessment
of player performance during a match . Experienced coaches tend to rely on their
intuition to identify which players are performing well and which are beginning to decline.
While this intuition is often valuable, it is inevitably based on heuristic judgments that may
overlook subtle patterns, contextual effects or interactions  between multiple aspects of
performance (Anderson & Sally, 2013; Memmert et al., 2017).
A data -driven approach enables a more objective, systematic and reproducible
evaluation of player performance . Football performance can be described across
multiple dimensions, including physical output (e.g. sprints, distance covered), technical
actions (e.g. shots or passes), tactical context (e.g. formation and playing position), and
behaviour under defensive pressure. Regardless of the analytical pers pective adopted,
football matches are ultimately decided by goals. Consequently, a fundamental problem
in football analytics is estimating the probability that a given player will score a goal
(Decroos et al., 2019).
You are therefore asked to bu ild a predictive model  to support coaches in better
assessing a player’s likelihood of scoring a goal over the remaining duration of the
game. To facilitate this task, a base dataset has been prepared containing performance
data for all players who participated in the UEFA Under-21 European Championship
held in Slovakia in 2025 . The dataset is structured to reflect the dynamic nature of a
football match, with player performance observed in consecutive 15-minute intervals,
referred to as checkpoints.
The base table  (players_quarters_final) is ready-for-use in your modelling , created
based on selected sources mentioned below,  and includes for each player and each
checkpoint rolling performance aggregates describing activity in the most recent 15
minutes (variables prefixed with last15_) as well as cumulative aggregates summarizing
activity from the start of the match up to  the analysed moment (variables prefixed with
cumul_). These aggregates capture, among other variables, information on shots taken
and sprints performed. Importantly, the dataset also contains a binary dependent

## Page 2

2

variable scored_after, indicating whether t he player scored a goal after the analysed
interval. In addition to individual performance measures, contextual information such as
playing position, team formation, home versus away status, and substitution timing is
included, as these factors may influen ce goal-scoring opportunities independently of the
player’s own actions (Memmert et al., 2017).
In addition, you are provided with four source and supplementary datasets that include
event-level data, providing more detailed descriptions of player behaviour. These include
data on individual shot events ( player_appearance_shot_limited), sprint and running
actions ( player_appearance_run), passing behavior ( player_appearance_pass), and
actions under defensive pressure (player_appearance_behaviour_under_pressure). The
combination of aggregated checkpoint data and fine -grained event data allows for a
comprehensive analysis that links micro-level actions  with macro -level outcomes
(Decroos et al., 2019).
The overarching research questions guiding this case is:
 RQ1: How well player’s on-field behaviour during a match can predict the probability that
they will score a goal later in the game (measured by balanced accuracy, AUC)?
 RQ2: Which aspects of a player ’s on -field behaviour during a match determine the
probability that they will score a goal later in the game?
To gain deeper insights, several more detailed research questions can be formulated
and addressed:
 RQ3: Is information about sprints and shots alone sufficient to accurately predict
the probability of a player scoring a goal?
 RQ4: Does incorporating passing data or information about player behaviour under
defensive pressure improve the predictive performance of goal-scoring models?
 RQ5: Does short -term performance (recent 15 -minute activity) or cumulative
performance up to a given  point in the match have stronger influence on the
probability of scoring a goal?
 RQ6: Does an increase in short -term performance intensity relative to a player’s
overall performance level increase the probability of scoring a goal?
 RQ7: Do external factors that are not directly controlled by the player influence the
probability of the player scoring a goal, and if so, which of them matter most?
Your task is therefore not only to develop a predictive model, but also to analyze,
compare, and interpret  the role of individual behaviour, temporal dynamics, and
contextual factors in goal scoring. The main aim is to generate insights that can
complement coaching intuition and support better in -game and post -game
decision-making.

## Page 3

3

Bibliography
Anderson, C., & Sally, D. (2013). The Numbers Game: Why Everything You Know About
Football Is Wrong. Penguin Books.
Bunker, R. P., & Thabtah, F. (2019). A machine learning framework for sport result
prediction. Applied Computing and Informatics, 15(1), 27–33.
Decroos, T., Bransen, L., Van Haaren, J., & Davis, J. (2019). Actions speak louder than
goals: Valuing player actions in soccer. Proceedings of the 25th ACM SIGKDD
International Conference on Knowledge Discovery & Data Mining, 1851–1861.
Memmert, D., Lemmink, K. A. P. M., & Sampaio, J. (2017). Current approaches to
tactical performance analyses in soccer. Sports Medicine, 47(1), 1–10.

Main dataset for the contest
 players_quarters_final.csv - the table contains player -level football match data split into 15 ‑minute
segments, showing participation details, positions, formations, and substitution information for each
player. It also records both 15 -minutes interval -based and cumulative physical  and attacking
performance metrics, such as distance covered, sprints, speed, shots, and goals. In the data set  the
scored_after column is a dependent variable.

Source and supplementary datasets for potential extension of feature space
 player_appearance_shot_limited.csv – table contains all shots taken during matches, including:
goals, shots on target, shots off target, blocked shots, penalties and own goals , but with NO outcome
of the shot included – to avoid information lea kage in the modelling process . Selected aggregates for
last 15 minutes and cumulatively from the beginning of a player occurrence are already included in the
base dataset.
 player_appearance_run.csv – table records high -intensity physical actions, specifically: high -speed
runs (hsr) and sprints (sprint).  Selected aggregates for last 15 minutes and cumulatively from the
beginning of a player occurrence are already included in the base dataset.
 player_appearance_pass.csv – table records every pass made during matches, with information
about: who made the pass, who the intended receiver was, whether the pass was accurate, when and
where it happened. Data from this table was NOT used to create a base dataset.
 player_appearance_behaviour_under_pressure.csv – single on-ball event that occurs while a player
is under defensive pressure (e.g. passes, ball carries, turnovers) captured during a match . Data from
this table was NOT used to create a base dataset.
