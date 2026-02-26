# Power BI Dashboard Walkthrough and Presentation Script

This document provides a professional page-by-page explanation of the Power BI dashboard used in the Student Wellness Early Warning and Triage project.

The dashboard is designed for screening and support prioritization. It is not a diagnosis dashboard.

## Before You Present

### One-line intro
This dashboard helps the university wellness team identify risk burden, understand the main drivers linked with higher risk, and convert those insights into an actionable support plan.

### Safety framing
Use this sentence early in the presentation:

> This is a risk screening and support prioritization tool. It does not make clinical diagnoses.

### Filter note for KPI consistency
The dashboard includes an outlier exclusion layer (age-degree consistency flag) used to stabilize KPI reporting. Because of this, some page totals may differ from the cleaned CSV row count.

- Cleaned CSV (`mental_health.csv`) rows: **27,853**
- Power BI Page KPI (with outlier filter logic): **27,197**

This difference is expected.

## Page 1 - Introduction

### Purpose of the page
This page sets the context and defines how to interpret the signals used in the dashboard.

### Suggested speaking script
Hi everyone. This dashboard is called the Student Mental Wellness Early Warning and Triage Dashboard.

Triage means deciding who needs support first based on urgency.

We built this because student mental health struggles often do not appear immediately in grades or attendance. Students may be dealing with stress, poor sleep, unhealthy routines, or emotional burnout, but they may not ask for help. By the time missed classes or visible crisis signals appear, early support may already be delayed.

This dashboard helps us act earlier by screening and prioritizing students using survey and lifestyle indicators, especially when counseling capacity is limited.

There are three things this dashboard helps us answer:
1. How many students look at risk right now
2. What factors are most strongly linked with that risk
3. Who should be reviewed first so outreach can be prioritized

This is not a diagnosis tool. It is a screening and support prioritization tool.

### Signal legend to explain on Page 1
- **Stress signals**: high academic pressure, financial stress, and related strain indicators
- **Routine signals**: sleep and diet patterns that are intervention-friendly and often easier to improve
- **Safety signals**: direct warning indicators such as suicidal thoughts that require faster attention and careful review
- **Triage layer**: the operational layer that combines signals into clear support priority tiers

## Page 2 - Executive Overview

### Purpose of the page
This page answers three executive questions:
- How big is the risk population?
- How serious is it?
- What workload does it create for the support team?

### Top KPI cards (what each means)
- **Students Screened**: total records included under current filters
- **Depressed Students**: count of `depression = 1`, with rate within the selected group
- **Suicidal Thoughts (Yes)**: count of students reporting suicidal thoughts; treated as a safety signal
- **Depressed + Suicidal**: overlap group that typically requires the fastest review

### Support Priority chart (most important operational visual)
This chart converts EDA signals into an action queue.

Priority tiers:
- **Critical**: highest urgency
- **High Priority**: urgent and should be contacted soon
- **Moderate Priority**: depressed students who still require review, but with lower red-flag intensity
- **Preventive High Risk**: not currently depressed, but strong stress/lifestyle risk signals
- **Preventive Watchlist**: early warning group to monitor and support early
- **Stable**: low-risk group, general wellness resources are usually sufficient

### How to explain "Preventive"
Preventive does not mean low importance. It means the student is not currently flagged as depressed, but the pattern of stress and lifestyle signals suggests elevated future risk. This supports early intervention before the case becomes urgent.

### Urgent vs Prevention workload split
The bottom-right workload split chart groups support priorities into two operational buckets:

- **Urgent** = Critical + High Priority
- **Prevention** = Preventive High Risk + Preventive Watchlist

This helps leadership quickly see how much work needs immediate follow-up versus proactive prevention support.

### Red flag ladder validation table (trust-building visual)
This table validates the engineered `red_flag` feature.

`red_flag` is a count of triggered conditions (0 to 7) based on:
- high academic pressure
- high financial stress
- long work or study hours
- low study satisfaction
- short sleep
- unhealthy diet
- suicidal thoughts

What this table should show:
- As `red_flag` increases, depression rate increases
- As `red_flag` increases, suicidal rate also tends to increase

This supports the triage logic by showing that the engineered risk ladder behaves meaningfully in the data.

### Slicers and interaction note
All values on this page should update when sliced by:
- gender
- age or age band
- degree category
- city

This allows leadership to compare risk burden and workload across segments without changing the underlying definitions.

## Page 3 - Drivers

### Purpose of the page
Page 2 answers "how much". Page 3 answers "why".

This page highlights the factors most strongly associated with higher depression risk in the dataset and identifies practical intervention levers.

### Headline rate cards
Keep two rates at the top for context:
- Overall depression rate
- Suicidal thoughts rate

These should remain slicer-aware so users can see how driver patterns differ by segment.

### Key takeaway cards (right side)
#### Top Risk Driver: Academic Pressure
Explain that academic pressure shows the strongest rise in depression rate across levels in the EDA.

Use the pattern summary:
- Low pressure levels are associated with lower depression rates
- Risk rises sharply at higher pressure levels

#### Top Protective Factor: Study Satisfaction
Explain that higher study satisfaction is associated with lower depression rates.

This acts as a protective factor in the EDA because depression rate tends to decrease as satisfaction increases.

### Driver ranking chart
This chart compares drivers using a gap-based measure (difference in depression rate between lower-risk and higher-risk groups or levels).

Interpretation guidance:
- A larger gap means a stronger association in this dataset
- It does not prove causation
- It helps prioritize intervention planning and messaging

Typical pattern in this project:
1. Academic pressure
2. Financial stress
3. Work or study hours
4. Diet
5. Study satisfaction (protective direction)
6. Sleep duration

### Bottom driver charts (how to explain each)
- **Academic Pressure -> Depression %**: generally increasing pattern from level 1 to level 5
- **Financial Stress -> Depression %**: usually increasing pattern with stress level
- **Study Satisfaction -> Depression %**: decreasing pattern as satisfaction rises
- **Dietary Habits -> Depression %**: unhealthy > moderate > healthy in many slices
- **Sleep Duration -> Depression %**: short sleep (especially less than 5 hours) tends to have the highest risk rates

### Page 3 summary line
Pressure and financial stress are the strongest risk-linked drivers in this dataset, study satisfaction looks protective, and sleep and diet are practical lifestyle levers for prevention-oriented interventions.

## Page 4 - Capacity and Action Plan

### Purpose of the page
This is the final decision page. It converts risk burden into an operational plan.

Pages 2 and 3 help us understand the problem. Page 4 helps leadership decide what to do next.

### Scenario controls (left side)
The page uses what-if controls to simulate staffing and throughput decisions:
- **Staff Count**
- **Follow-ups per Staff per Month**
- **Target Months to Clear Urgent**

### What the page calculates
1. **Monthly capacity**
   - `staff_count * follow_ups_per_staff_per_month`
2. **Urgent backlog**
   - Critical + High Priority
3. **Prevention backlog**
   - Preventive High Risk + Preventive Watchlist
4. **Months to clear urgent**
   - Urgent backlog / monthly capacity (scenario-based)
5. **Required capacity to meet target**
   - Urgent backlog / target months
6. **Capacity gap**
   - Required capacity - current capacity
7. **Additional staff needed**
   - Capacity gap / follow-ups per staff (rounded appropriately)

### Example narration (based on the project scenario)
Using the current scenario settings:
- Staff Count = 48
- Follow-ups per staff per month = 10
- Target months to clear urgent = 3

Current monthly capacity = **480 follow-ups**.

If the urgent backlog is **4,103**, then clearing urgent cases first with 480 follow-ups per month takes about **9 months**.

To meet a 3-month urgent clearance target, required monthly capacity is about **1,368**. Compared with current capacity (480), the gap is about **888 follow-ups per month**.

At 10 follow-ups per staff per month, this implies about **89 additional staff** for that target scenario.

### Decision message for leadership
This page turns analytics into a staffing conversation:
- Do we increase staff?
- Do we increase follow-ups per staff?
- Do we accept a longer timeline?
- Do we focus only on the highest-risk subgroup first?

## Recommended Final Closing (Presentation)

This dashboard turns survey data into a practical action plan.

Page 2 shows the size of the risk burden and triage workload.
Page 3 explains the strongest factors linked with that risk.
Page 4 converts those insights into capacity planning, timeline estimates, and staffing scenarios.

The result is a screening and prioritization workflow that helps support teams act earlier and plan realistically, while staying within a clear non-diagnostic boundary.

## Beginner Tips for Using the Dashboard

- Start with Page 2 to understand the current burden before slicing into subgroups.
- Use one slicer at a time first (for example, degree category) to avoid over-filtering.
- Check the red flag ladder table when validating whether a slice still has a meaningful risk pattern.
- Move to Page 3 after you identify a high-risk segment to understand likely drivers.
- Use Page 4 last to test staffing scenarios and target timelines.
- Reset slicers before presenting a new segment to avoid confusion.
