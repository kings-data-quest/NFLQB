# QB Metric Correlation Analysis Report

## Correlation with Team Success

### Correlation Coefficients

|                    |   team_win_pct |       wins |   points_for |
|:-------------------|---------------:|-----------:|-------------:|
| qb_composite_score |       0.335583 |   0.335583 |     0.356515 |
| passer_rating      |     nan        | nan        |   nan        |
| qbr                |     nan        | nan        |   nan        |

### Statistical Significance (p-values)

|                    |   team_win_pct |       wins |   points_for |
|:-------------------|---------------:|-----------:|-------------:|
| qb_composite_score |       0.019713 |   0.019713 |    0.0128744 |
| passer_rating      |     nan        | nan        |  nan         |
| qbr                |     nan        | nan        |  nan         |

## Interpretation

The metric with the strongest correlation to team wins is **qb_composite_score** with a correlation coefficient of **0.336**.

### Comparison to Traditional Metrics

Our custom QB composite score has a correlation of **0.336** with team wins.

This is **0.336 higher** than the correlation of passer_rating (0.000).

This is **0.336 higher** than the correlation of qbr (0.000).

## QB Rating Discrepancies

No significant discrepancies were found between rating systems.

## Conclusion

This analysis demonstrates the relationship between quarterback performance metrics and team success. The custom QB composite metric developed in this project incorporates multiple aspects of quarterback play including accuracy at different depths, performance under pressure, and mobility contribution.

The results show that our custom QB metric has a moderate positive correlation with team success.

Further research could explore additional factors and refine the weighting of different components in the composite metric to better predict team success.