# Requirements Document

## Introduction

The Coach Pass Rush Assistant is a web application for NFL coaching staffs built on Big Data Bowl data (plays.csv, pffScoutingData.csv, with players.csv and games.csv as support). It estimates, for a given pass-play situation, the probability of pressure or sack and the expected yards gained, and recommends the call (for example coverage, number of pass rushers, or blitz) with the best predicted outcome. Exploratory analysis shows pressure on 37.6% of plays, with 3.94 yards per play under pressure versus 8.15 without, and a sack rate of 17% under pressure versus 0% without.

The application supports two modes: Game Replay (the coach selects a past game and steps through plays, seeing a recommendation at each snap) and What-If (the coach enters down, distance, formation, and expected coverage). It also provides a pre-game Opponent Report. Predictions use two chained linear regressions. Everything is computed in memory at startup or at response time on an ephemeral server, and the source data is read-only. The MVP excludes tracking data (the large week files are not loaded). Amazon Nova is an optional extension and is not required for any core feature.

## Glossary

- **Assistant**: The complete Coach Pass Rush Assistant application (backend and dashboard).
- **Backend**: The FastAPI application exposed from backend/main.py that serves the API under /api/.
- **Dashboard**: The React, Vite, Recharts, and TypeScript frontend located in dashboard/src.
- **Data_Loader**: The Backend component that reads and cleans the source CSV files.
- **Data_Directory**: The folder holding the source CSV files, resolved from a relative path configured through an environment variable and defaulting to Datos/data/.
- **Pressure_Model**: The linear regression that predicts the probability of pressure or sack for a play situation.
- **Yards_Model**: The linear regression that predicts expected yards and takes the Pressure_Model output as an input feature.
- **Recommender**: The Backend component that evaluates candidate defensive calls with the Pressure_Model and Yards_Model and selects the best predicted outcome.
- **Candidate_Call**: A defensive configuration evaluated by the Recommender, defined by coverage, number of pass rushers, and defenders in the box.
- **Best_Call**: The Candidate_Call with the highest Pressure_Model probability among those evaluated; ties are resolved by the lowest Yards_Model expected yards.
- **Play_Situation**: The set of inputs describing a snap: down, yards to go, offense formation, expected coverage, and optional context fields.
- **Replay_Mode**: The mode in which the coach selects a past game and steps through its plays.
- **WhatIf_Mode**: The mode in which the coach manually enters a Play_Situation.
- **Opponent_Report**: The pre-game summary of a selected team's pass-protection and pass-rush tendencies.
- **Pass_Play**: A play in plays.csv that has at least one pffScoutingData row with pff_role equal to Pass Rush or Pass Block.
- **Pressure_Event**: A Pass_Play in which at least one pffScoutingData row has a nonzero value in a pressure or sack column (hit, hurry, or sack).
- **Optional_AI_Service**: The optional Amazon Nova model, invoked through Bedrock using the BEDROCK_MODEL_ID environment variable.
- **Stack_Contract**: The hackathon rules in .hackathon/rules.md and .kiro/steering/rules.md.

## Requirements

### Requirement 1: Data Loading and Configuration

**User Story:** As a developer, I want the Backend to load the source data from a configurable location, so that the application runs in any environment without moving the data files.

#### Acceptance Criteria

1. WHEN the Backend starts, THE Data_Loader SHALL read plays.csv, pffScoutingData.csv, players.csv, and games.csv from the Data_Directory and SHALL load each file completely into memory before the Backend reports readiness.
2. WHERE the Data_Directory environment variable is set to a non-empty value, THE Data_Loader SHALL resolve the Data_Directory as that path relative to the project root, and OTHERWISE THE Data_Loader SHALL use Datos/data/ relative to the project root.
3. IF one or more required CSV files are missing from the Data_Directory, THEN THE Backend SHALL abort startup with a non-zero exit status and SHALL report the name of every missing file in the startup error message.
4. IF startup aborted because of a missing CSV file, THEN THE Backend SHALL, for every request to a path beginning with /api/, return an error response that lists the name of every missing file.
5. IF a required CSV file exists but cannot be read or parsed (unreadable, empty, or missing its header row), THEN THE Backend SHALL abort startup with a non-zero exit status and SHALL report the name of the affected file and the reason category (unreadable, empty, or malformed) in the startup error message.
6. THE Data_Loader SHALL open all source files in read-only mode and SHALL NOT create, modify, or delete any file in the Data_Directory.
7. THE Data_Loader SHALL NOT open any tracking CSV file during startup or request handling.
8. THE Backend SHALL keep all loaded and derived data in memory and SHALL NOT require write access to the file system for the application to operate.
9. WHEN the Backend receives GET /health, THE Backend SHALL respond with the JSON body {"status": "ok"} within 500 milliseconds.
10. WHEN the Data_Loader finishes loading, THE Backend SHALL be ready to serve /api/ requests no later than 60 seconds after the Backend process starts, on a machine running Python 3.12.

### Requirement 2: Data Cleaning and Edge Cases

**User Story:** As a data analyst, I want the known data irregularities handled consistently, so that the models are trained and queried on valid features.

#### Acceptance Criteria

1. WHEN a play has down equal to 0, THE Data_Loader SHALL label the play as a two-point conversion, keep the play in the dataset, and leave the down value of 0 unchanged in the output row.
2. WHEN a play has quarter equal to 5, THE Data_Loader SHALL label the play as overtime, keep the play in the dataset, and leave the quarter value of 5 unchanged in the output row.
3. WHEN a play has a null dropBackType or offenseFormation, THE Data_Loader SHALL replace the null with the category "Unknown".
4. WHEN a play has a null defendersInBox, THE Data_Loader SHALL replace the null with the median defendersInBox value computed only from the training-set plays that have a non-null defendersInBox.
5. THE Data_Loader SHALL match pff_role values using the exact case-sensitive strings "Pass Block", "Pass Route", and "Pass Rush".
6. THE Data_Loader SHALL compute blocking metrics only from pffScoutingData rows whose pff_role is "Pass Block", and SHALL compute rushing metrics only from rows whose pff_role is "Pass Rush".
7. WHEN a pffScoutingData row has a null in a field that applies only to a different pff_role, THE Data_Loader SHALL treat the null as not applicable and SHALL exclude it from the count of missing values and from imputation.
8. WHEN the Data_Loader joins plays.csv and pffScoutingData.csv, THE Data_Loader SHALL join on gameId and playId together as a composite key, and SHALL NOT join on either identifier alone.
9. WHEN the Data_Loader builds the play-level dataset, THE Data_Loader SHALL derive exactly one row per Pass_Play, with a Pressure_Event label of 0 or 1 and the yards gained as a numeric value.
10. IF a required input file is missing, unreadable, or lacks a column needed for the joins or labels above, THEN THE Data_Loader SHALL stop without producing a dataset, SHALL report an error identifying the file or column, and SHALL leave any previously produced output unmodified.

### Requirement 3: Pressure Probability Model

**User Story:** As a coach, I want an estimate of the probability of pressure or sack for a situation, so that I can judge how effective a defensive call will be.

#### Acceptance Criteria

1. WHEN the Backend starts, THE Pressure_Model SHALL be fitted in memory as a linear regression on the play-level dataset with the Pressure_Event label as the target, and THE Backend SHALL complete the fit within 30 seconds.
2. THE Pressure_Model SHALL use down, yards to go, offense formation, expected coverage, number of pass rushers, and defendersInBox as input features.
3. WHEN the Pressure_Model receives a Play_Situation, THE Pressure_Model SHALL return a pressure probability as a number clipped to the range 0 to 1 inclusive.
4. WHEN a client requests the Pressure_Model fit metrics through the /api/ endpoint, THE Backend SHALL return the R-squared and the mean absolute error computed on the held-out split, together with the number of rows in the training split and in the held-out split.
5. THE Backend SHALL split the play-level dataset into 80% training rows and 20% held-out rows using a fixed random seed, so that the reported metrics are identical across restarts on the same dataset.
6. IF the Play_Situation has a missing input feature, a feature value outside the values seen in the training data, or a down outside 0 to 4, THEN THE Pressure_Model SHALL reject the request with an error indication that names the invalid feature, and THE Backend SHALL leave the fitted model unchanged.
7. IF the fit fails at startup because the dataset is missing, is empty, or has fewer than 100 labeled plays, THEN THE Backend SHALL report the failure in the startup output, and THE Backend SHALL answer Pressure_Model requests with an error indication that the model is unavailable.

### Requirement 4: Expected Yards Model

**User Story:** As a coach, I want an estimate of expected yards for a situation, so that I can compare calls by their outcome on the field.

#### Acceptance Criteria

1. WHEN the Backend starts, THE Yards_Model SHALL be fitted in memory as a linear regression on the play-level dataset with yards gained as the target, using the same 80/20 train and held-out split and fixed random seed as the Pressure_Model.
2. THE Yards_Model SHALL use the Pressure_Model output (a probability from 0.0 to 1.0) together with down (integer 0 to 4), yards to go (integer 0 to 99), offense formation, and expected coverage as input features.
3. WHEN the Yards_Model receives a valid Play_Situation, THE Yards_Model SHALL obtain the pressure probability from the Pressure_Model and SHALL return the expected yards as a number rounded to 2 decimal places.
4. IF the Yards_Model receives a Play_Situation with a missing field, a down outside 0 to 4, a yards to go outside 0 to 99, or an offense formation or expected coverage value not seen in the training data, THEN THE Yards_Model SHALL reject the request, SHALL return an error indication that names the invalid field, and SHALL leave the fitted model unchanged.
5. THE Backend SHALL expose the fit metrics of the Yards_Model (R-squared and mean absolute error in yards, both computed on the held-out split) through an /api/ endpoint that responds within 500 ms.
6. IF the Yards_Model fitting fails at startup, THEN THE Backend SHALL report the failure in an error indication and SHALL respond to Yards_Model requests and to the metrics endpoint with an error indication stating that the model is unavailable.
7. WHILE the Pressure_Model output varies and all other features are held constant, THE Yards_Model SHALL return expected yards that change by the fitted pressure coefficient multiplied by the change in pressure probability, within a tolerance of 0.01 yards.

### Requirement 5: Call Recommendation

**User Story:** As a coach, I want the application to recommend the call with the best predicted outcome, so that I can make faster and better-informed decisions.

#### Acceptance Criteria

1. WHEN the Backend receives a valid Play_Situation, THE Recommender SHALL evaluate every Candidate_Call observed in the training data and return the Best_Call, defined as the Candidate_Call ranked first under criterion 3.
2. THE Recommender SHALL return, for the Best_Call and for each other evaluated Candidate_Call, the predicted pressure probability (a value from 0.0 to 1.0 inclusive) and the predicted expected yards (rounded to one decimal place).
3. THE Recommender SHALL return the Candidate_Calls ranked from best to worst, ordered by predicted pressure probability in descending order. IF two Candidate_Calls have equal predicted pressure probability, THEN THE Recommender SHALL rank the one with the lower predicted expected yards first.
4. WHEN the Recommender returns a result, THE Recommender SHALL include the predicted difference in expected yards (Best_Call minus typical call, in yards, rounded to one decimal place) between the Best_Call and the Candidate_Call labeled as the typical call for that Play_Situation. The typical call is the Candidate_Call with the highest play count among the training-data plays matching that Play_Situation.
5. IF a Play_Situation contains a down equal to 0, THEN THE Recommender SHALL return the recommendation with an indicator labeling it as a two-point conversion situation, and THE Recommender SHALL still return the ranked Candidate_Calls defined in criteria 1 to 4.
6. IF the number of training-data plays matching a Play_Situation is fewer than 30, THEN THE Recommender SHALL mark the recommendation as low-confidence and still return the ranked Candidate_Calls.
7. IF the Backend receives a Play_Situation with a missing required field or a field value outside its allowed range, or IF the Recommender cannot produce predictions, THEN THE Recommender SHALL return an error response indicating the cause (invalid input or prediction unavailable) within 2 seconds and SHALL NOT return a Best_Call.

### Requirement 6: What-If Mode

**User Story:** As a coach, I want to enter a hypothetical situation manually, so that I can plan for scenarios before they happen.

#### Acceptance Criteria

1. WHILE WhatIf_Mode is active, THE Dashboard SHALL display input controls for down (integer), yards to go (integer), offense formation (single selection), and expected coverage (single selection), and SHALL display a submit control that is disabled until all four inputs have a value.
2. WHEN the coach submits a complete Play_Situation in WhatIf_Mode, THE Dashboard SHALL display, within 3 seconds of the Backend response being received, the Best_Call and the Candidate_Calls in the order returned by the Recommender. For each Candidate_Call it SHALL show the predicted pressure probability (as a percentage from 0% to 100%) and the expected yards (to one decimal place).
3. IF the coach enters a value outside the valid range (down: integer from 0 to 4; yards to go: integer from 0 to 99) or leaves a field empty or non-numeric, THEN THE Dashboard SHALL display a validation message naming each invalid field, SHALL NOT send the request to the Backend, and SHALL retain all values the coach entered.
4. IF the Backend receives a Play_Situation with a missing field, a value outside the valid range, or a formation or coverage value not present in the training data, THEN THE Backend SHALL reject the request with HTTP status 422, SHALL return a message naming each invalid field, and SHALL NOT compute predictions for that request.
5. THE Dashboard SHALL list as selectable options for offense formation and expected coverage only the distinct values present in the training data, with no free-text entry permitted for these two fields.
6. WHEN the coach changes any input in WhatIf_Mode after a result is shown, THE Dashboard SHALL, within 500 milliseconds, mark the displayed result as outdated, SHALL keep the previous result visible, and SHALL remove the outdated mark only after the coach submits again and a new result is displayed.
7. IF the Backend does not respond within 10 seconds or returns an error other than a validation rejection, THEN THE Dashboard SHALL display an error message indicating the request failed, SHALL retain the coach's entered values, and SHALL keep the submit control enabled for retry.

### Requirement 7: Game Replay Mode

**User Story:** As a coach, I want to step through a past game play by play with a recommendation at each snap, so that I can compare the recommended call with what the defense actually did.

#### Acceptance Criteria

1. WHEN a client requests the list of games, THE Backend SHALL return through an /api/ endpoint every game in games.csv, each with its date, home team, and visitor team, ordered by date ascending.
2. WHEN the coach selects a game in Replay_Mode, THE Dashboard SHALL display the Pass_Plays of the game ordered by quarter ascending and then by game clock descending (from 15:00 to 00:00) within each quarter, and SHALL display the first play within 2 seconds of selection.
3. WHEN the coach steps to a play in Replay_Mode, THE Dashboard SHALL display the quarter, down, yards to go, formation, coverage, the Best_Call, the actual call used, the actual pressure outcome, and the actual yards gained, within 1 second of the step action.
4. WHEN the coach steps to a play with quarter equal to 5, THE Dashboard SHALL display the quarter as "overtime" instead of a numeric quarter.
5. WHEN the coach steps to a play with down equal to 0, THE Dashboard SHALL display the play as a two-point conversion and SHALL NOT display a numeric down.
6. THE Dashboard SHALL provide a step forward control, a step backward control, and a jump control that lets the coach choose any play of the selected game by its position number from 1 to the total number of Pass_Plays.
7. WHILE the first play of the game is displayed, THE Dashboard SHALL disable the step backward control.
8. WHILE the last play of the game is displayed, THE Dashboard SHALL disable the step forward control.
9. IF the selected game has no Pass_Plays, THEN THE Dashboard SHALL display a message stating that the game has no pass plays available, SHALL disable the step forward, step backward, and jump controls, and SHALL NOT display the cumulative comparison.
10. WHEN the coach steps or jumps to a play, THE Dashboard SHALL display a cumulative comparison, over plays 1 through the displayed play inclusive, of the sum of actual yards gained versus the sum of predicted yards of the Best_Call, and the difference between the two sums, in yards rounded to one decimal place.
11. IF the Backend request for the game list or for the Pass_Plays of a selected game fails or does not respond within 10 seconds, THEN THE Dashboard SHALL display an error message indicating that the replay data could not be loaded, SHALL keep the previously displayed game and play unchanged, and SHALL provide a retry control.

### Requirement 8: Opponent Report

**User Story:** As a coach, I want a pre-game report on an opponent, so that I can prepare based on their pass protection and pass rush tendencies.

#### Acceptance Criteria

1. WHEN the coach selects a team that has at least one Pass_Play in the dataset, THE Backend SHALL return the Opponent_Report containing, for that team on offense and separately on defense, the number of Pass_Plays (integer of 0 or more), the pressure rate (decimal from 0.0 to 1.0, rounded to 3 decimal places), the sack rate (decimal from 0.0 to 1.0, rounded to 3 decimal places), and the average yards per play (rounded to 2 decimal places).
2. WHEN the coach selects a team that has at least one Pass_Play in the dataset, THE Backend SHALL return the pressure rate and the average yards per play of that team, each broken down by coverage type and by offense formation, and each breakdown group SHALL include its Pass_Play count.
3. WHEN the coach selects a team that has at least one Pass_Play in the dataset, THE Backend SHALL return the top five players of that team ranked by total sacks in descending order and the top five players ranked by total pressures in descending order, each with player name and position taken from players.csv.
4. IF fewer than five players of the selected team have at least one sack or pressure, THEN THE Backend SHALL return only the players with at least one, in descending order, without padding the list.
5. IF two or more players have the same count at the fifth rank, THEN THE Backend SHALL break the tie by player name in ascending alphabetical order so the same input always yields the same list.
6. IF a player in the ranking has no matching record in players.csv, THEN THE Backend SHALL include that player with the player identifier in place of the name and with the position shown as unknown.
7. WHEN the Backend returns an Opponent_Report, THE Dashboard SHALL present it with Recharts charts that compare the selected team's pressure rate, sack rate, and average yards per play against the league average, with the selected team and the league average each shown as a distinct labeled series.
8. IF the coach selects a team with no Pass_Plays in the dataset, THEN THE Backend SHALL return an empty report and a message indicating that no data is available for that team, and THE Dashboard SHALL display that message in place of the charts.
9. IF the Backend fails to compute the Opponent_Report, THEN THE Backend SHALL return an error indication with a message stating that the report could not be generated, and THE Dashboard SHALL display that message and retain the previously selected team.
10. THE Backend SHALL compute the pressure rate for the Opponent_Report using the same Pressure_Event definition as the Pressure_Model target.

### Requirement 9: Exploratory Findings Display

**User Story:** As a coach, I want to see the key pressure findings from the data, so that I understand why the recommendations matter.

#### Acceptance Criteria

1. WHEN the Dashboard findings view loads, THE Dashboard SHALL display five summary values: the overall pressure rate (percentage of Pass_Plays with pressure), the average yards per play with pressure, the average yards per play without pressure, the sack rate with pressure, and the sack rate without pressure, with each value shown to one decimal place and with its unit (% or yards).
2. WHEN the Backend starts, THE Backend SHALL compute every value shown under criterion 1 from the loaded data and SHALL NOT use hard-coded values.
3. IF the loaded data contains zero plays in the denominator of a summary value (for example, no plays with pressure), THEN THE Dashboard SHALL display "N/A" for that value and SHALL NOT display NaN, Infinity, or a blank.
4. WHEN the Dashboard findings view loads, THE Dashboard SHALL display three Recharts charts, each showing pressure rate (%) and average yards per play for the groups of one dimension: coverage, offense formation, and defendersInBox count, with every group in the loaded data shown and each group labeled with its name or count.
5. WHEN the Dashboard findings view loads, THE Dashboard SHALL display a ranking of the top 10 players by total sacks and a ranking of the top 10 teams by pressure rate, each in descending order, showing the player or team name and the ranked value.
6. IF the Dashboard fails to retrieve findings data from the Backend, or the response is empty, THEN THE Dashboard SHALL display an error message indicating that findings are unavailable, SHALL display no partial or placeholder values, and SHALL provide a retry control.

### Requirement 10: Dashboard Delivery and Stack Compliance

**User Story:** As a hackathon participant, I want the application to comply with the Stack_Contract, so that the hosting platform deploys it unchanged.

#### Acceptance Criteria

1. THE Backend SHALL expose the FastAPI application as a module-level variable named app in backend/main.py and SHALL register every API route under the /api/ prefix, with zero routes outside /api/ other than /health and the static dashboard/dist file serving defined in criterion 4.
2. THE Dashboard SHALL call the Backend using relative paths without a leading slash (for example fetch('api/<route>')), with zero request URLs that begin with "/" or contain a scheme and host.
3. WHEN the command vite build runs in the dashboard folder, THE Dashboard SHALL complete with exit code 0, write the output to dashboard/dist, and keep the Vite base setting equal to './'.
4. WHEN a client requests the root URL, THE Backend SHALL return the dashboard/dist index page and serve both the static assets and the /api/ routes from the same host and the same single port.
5. THE Assistant SHALL use only the dependency versions defined in the Stack_Contract (Python 3.12 in .venv, pandas 2.2.3, numpy 2.1.3, FastAPI, React, Vite, Recharts, TypeScript, Tailwind CSS 3), and SHALL add no package that is absent from the Stack_Contract.
6. THE Assistant SHALL implement both linear regressions using only numpy and pandas, with zero machine learning libraries (for example scikit-learn, statsmodels, scipy, TensorFlow, PyTorch) added to the dependencies.
7. THE Assistant SHALL leave the .hackathon/, .venv/, and .kiro/ folders and the existing top-level folder structure with no files or folders renamed, moved, or deleted.
8. WHILE a request to the Backend is pending, THE Dashboard SHALL display a loading indicator in the view that depends on that request, and SHALL remove it within 500 ms of the response or failure.
9. IF a request to the Backend fails (network error, non-success response, or no response within 10 seconds), THEN THE Dashboard SHALL display in the affected view an error message indicating that the data could not be loaded, SHALL provide a retry control, and SHALL keep the other views usable.

### Requirement 11: Optional Amazon Nova Extension

**User Story:** As a coach, I want an optional plain-language explanation of a recommendation, so that I can communicate the reasoning to my staff.

#### Acceptance Criteria

1. WHERE the Optional_AI_Service is available, WHEN a client requests an explanation for an existing recommendation, THE Backend SHALL return a plain-language explanation of at most 1,000 characters that references the recommended play, its probability, and its expected yards, within 10 seconds of the request.
2. WHERE the Optional_AI_Service is available, THE Backend SHALL create the Bedrock client with only the region from the AWS_REGION environment variable and SHALL read the model from the BEDROCK_MODEL_ID environment variable.
3. IF the Optional_AI_Service is unavailable, returns an error, or does not respond within 10 seconds, THEN THE Backend SHALL respond to the explanation request with a failure response containing a message indicating that the explanation is unavailable, SHALL leave the stored recommendation unchanged, and SHALL continue to serve all other endpoints.
4. THE Backend SHALL NOT read, verify, or log AWS credentials, and SHALL NOT include credentials in source code.
5. THE Assistant SHALL produce all recommendations, probabilities, and expected yards without invoking the Optional_AI_Service.
6. IF the explanation request refers to a recommendation that does not exist, THEN THE Backend SHALL respond with an error indicating that the recommendation was not found and SHALL NOT invoke the Optional_AI_Service.
7. IF the AWS_REGION or BEDROCK_MODEL_ID environment variable is missing or empty, THEN THE Backend SHALL respond to the explanation request with a message indicating that the explanation is unavailable and SHALL continue to serve all other endpoints.