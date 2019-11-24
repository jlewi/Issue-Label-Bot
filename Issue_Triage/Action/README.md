# Triage Issues For Kubeflow
A GitHub Action that triages issues for the [Kubeflow](https://github.com/kubeflow/) project.

## Discussion

- [Notebook](https://github.com/kubeflow/code-intelligence/blob/master/Issue_Triage/notebooks/triage.ipynb) exploring ways to programatically triage issues.
- [Issue](https://github.com/kubeflow/community/issues/278) describing requirements for automatic triage.

# Usage

## Example Workflow

```yaml
name: Check Triage Status of Issue
on: 
  issues:
    types: [opened, closed, reopened, transferred, labeled, unlabeled]
    # Issue is created, Issue is closed, Issue added or removed from projects, Labels added/removed

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Update Kanban
        uses: kubeflow/triage-issues@master
        with:
          PROJECT_CARD_ID: 'MDEzOlByb2plY3RDb2x1bW41OTM0MzEz'
          ISSUE_NUMBER: ${{ github.event.issue.number }}
          PERSONAL_ACCESS_TOKEN: ${{ secrets.triage_projects_github_token }}
```

## Mandatory Inputs

1. **PROJECT_CARD_ID**: The Project Card ID that you want to move issues to.  Defaults to `MDEzOlByb2plY3RDb2x1bW41OTM0MzEz`
2. **ISSUE_NUMBER**: The issue number in the current repo that you want to triage
3. **PERSONAL_ACCESS_TOKEN**: A personal access token with authorization to modify the project board.

