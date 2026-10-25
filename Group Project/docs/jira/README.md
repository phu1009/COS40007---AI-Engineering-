# Importing the backlog into Jira

`jira_import.csv` holds the full backlog: 6 epics, 7 user stories (US1–US7) and 23 tasks (T1–T23) as sub-tasks, linked through the `Issue ID` and `Parent` columns.

1. Create a Jira Software project: **Scrum**, **company-managed**, key **ASL**.
2. Go to **Settings (⚙) → System → External system import → CSV**. If Jira offers it, choose "Switch to the old experience". Upload `jira_import.csv` and choose the ASL project.
3. Map the columns:
   - Issue ID → Issue Id
   - Issue Type → Issue Type
   - Summary → Summary
   - Parent → Parent
   - Description → Description
   - Priority → Priority
   - Story Points → Story point estimate (or Story Points)
   - Labels → Labels
4. Run the import. If Jira says the issue type `Sub-task` does not exist (team-managed projects call it `Subtask`), replace `Sub-task` with `Subtask` in the CSV and import again.
5. In the backlog, create sprints **S0–S4** and set their dates from the Canvas teaching calendar. Filter by the labels `sprint-S1` … `sprint-S4` and drag the stories into each sprint.
6. Assign owners with the filters `labels = owner-M1` and `labels = owner-M2`.
7. Take two screenshots for the report: the backlog view (Figure 2) and the sprint board (Figure 3).
