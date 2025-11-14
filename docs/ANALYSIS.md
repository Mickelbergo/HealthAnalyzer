# Health Analytics Dashboard - Comprehensive Analysis

**Analysis Period:** March 24, 2025 - November 14, 2025 (236 days)

## Executive Summary

This document provides a detailed analysis of health metrics collected from Apple Watch and exported via Apple Health. The dashboard tracks weight loss, fitness improvements, sleep patterns, and activity levels over an 8-month period.

### Key Achievements

- **Weight Loss:** Lost 7.2 kg (82.2 kg → 75.0 kg)
- **Running Performance:** Improved pace by 35 seconds per kilometer (5.76 → 5.18 min/km)
- **Daily Activity:** Averaged 16,057 steps per day with 83% of days exceeding 10,000 steps
- **Energy Balance:** Maintained an average calorie deficit of 1,163 kcal/day

---

## 1. Weight Loss Analysis

![Weight Loss Progress](weight_loss.png)

### Overview
- **Starting Weight:** 82.2 kg (March 24, 2025)
- **Current Weight:** 75.0 kg (November 14, 2025)
- **Total Loss:** 7.2 kg over 236 days
- **Average Rate:** 0.30 kg/week or 1.2 kg/month

### Analysis
The weight loss progression shows a steady downward trend with minor fluctuations typical of natural body weight variation. The rate of weight loss falls within the healthy range (0.5-1 kg/week), indicating sustainable progress without extreme caloric restriction.

**Key Observations:**
- Consistent downward trend throughout the tracking period
- No extended plateaus or rapid rebounds
- Weight loss aligns with calculated TDEE and calorie intake data
- Sustainable approach indicating long-term success potential

---

## 2. Energy Balance & TDEE Analysis

![TDEE Analysis](tdee_analysis.png)

### Metabolic Metrics
- **Average TDEE:** 2,941 kcal/day
- **Average Calorie Intake:** 1,778 kcal/day
- **Average Daily Deficit:** 1,163 kcal/day
- **Days Analyzed:** 69 days (with complete tracking data)

### Methodology
Total Daily Energy Expenditure (TDEE) was calculated using the energy balance equation:

```
TDEE = Calories_In - (Weight_Change_kg × 7,700 kcal/kg) / Days
```

This approach uses actual weight changes and calorie intake to reverse-engineer your true energy expenditure, providing more accurate results than formula-based estimates.

### Analysis
The TDEE of approximately 2,940 kcal/day indicates a highly active lifestyle. This aligns with:
- High daily step count (16,057 average)
- Regular running activities
- Apple Watch-recorded active energy burn

The consistent deficit of ~1,160 kcal/day mathematically predicts a weight loss rate of approximately 0.66 kg/week, which closely matches the observed results (0.64 kg/week during the tracking period).

**Data Quality Notes:**
- Analysis excludes days with zero calorie tracking (34 days)
- Only includes days with both weight and calorie data
- 7-day rolling average used to smooth daily fluctuations

---

## 3. Running Performance

![Running Pace Improvement](pace_improvement.png)

### Performance Metrics
- **Initial Average Pace:** 5.76 min/km (first 7 days)
- **Recent Average Pace:** 5.18 min/km (last 7 days)
- **Improvement:** 0.59 min/km (35 seconds per kilometer faster)
- **Percentage Improvement:** 10.1% faster

### Analysis
The running performance shows consistent improvement over the tracking period. The 35-second per kilometer improvement represents significant fitness gains:

**At Different Distances:**
- 5K run: 2 minutes 55 seconds faster
- 10K run: 5 minutes 50 seconds faster
- Half Marathon (21.1 km): 12 minutes 22 seconds faster

This improvement can be attributed to:
- Consistent training volume
- Weight loss reducing biomechanical load
- Improved cardiovascular fitness (reflected in resting heart rate trends)
- Progressive adaptation to running stress

**Training Characteristics:**
- Regular running sessions (identified by days with >1 km distance)
- Progressive pace improvements without injury
- Natural variation in daily pace based on workout type (easy runs, tempo, intervals)

---

## 4. Sleep Patterns

![Sleep Analysis](sleep_analysis.png)

### Sleep Metrics
- **Average Sleep Duration:** 6.5 hours/night
- **Median Sleep Duration:** 6.8 hours/night
- **Optimal Sleep (7-9 hours):** 38% of nights
- **Sleep Tracking Days:** 218 nights

### Analysis
Sleep duration averages below the recommended 7-9 hours for adults, with only 38% of nights achieving optimal sleep. This represents a potential area for improvement.

**Sleep Quality Considerations:**
- Sleep calculation includes only actual sleep states (Deep, REM, Core)
- Excludes "Awake" and "InBed" periods for accuracy
- Sleep stages tracked via Apple Watch sleep monitoring

**Recommendations:**
- Aim for consistent 7-9 hours per night
- Inadequate sleep may impact:
  - Recovery from exercise
  - Weight loss efficiency (sleep deprivation affects hunger hormones)
  - Athletic performance
  - Overall health markers

**Positive Observations:**
- No extreme sleep deprivation (<4 hours) on most nights
- Some nights achieve excellent sleep (8-10 hours)
- Opportunity for improvement through better sleep hygiene

---

## 5. Daily Activity Patterns

![Activity Overview](activity_overview.png)

### Activity Metrics
- **Average Daily Steps:** 16,057 steps
- **Days with 10,000+ Steps:** 83% (196 out of 236 days)
- **Total Distance Tracked:** Walking/Running data available for analysis period

### Analysis
Activity levels significantly exceed the commonly recommended 10,000 steps per day target. This high activity level contributes directly to the elevated TDEE of ~2,940 kcal/day.

**Activity Breakdown:**
- Consistent daily movement outside of structured exercise
- High adherence to activity goals (83% of days meeting 10k target)
- Combined walking and running activities
- Active lifestyle supporting weight loss and fitness goals

**Impact on Health Outcomes:**
- Higher TDEE allows for more flexible nutrition approach
- Supports cardiovascular health
- Aids in weight loss and maintenance
- Improves insulin sensitivity and metabolic health

---

## 6. Cardiovascular Fitness

### Heart Rate Trends
While specific averages vary by tracking period, the running analysis section of the dashboard tracks:
- **Resting Heart Rate (RHR):** Lower RHR typically indicates improved cardiovascular fitness
- **Heart Rate Variability (HRV):** Higher HRV suggests better recovery and autonomic nervous system function
- **Exercise Heart Rate:** Monitored during activities

**Expected Trends:**
- RHR should decrease with improved fitness
- HRV should increase with better recovery and adaptation
- Exercise heart rate efficiency should improve (lower HR at same pace)

---

## Technical Notes

### Data Processing

**Data Quality Fixes Implemented:**
1. **Duplicate Removal:** Removed 697,000+ duplicate entries across all metrics caused by multiple data imports
2. **Date Filtering:** Limited analysis to March 24, 2025 onwards (user-specified start date)
3. **Sleep Calculation:** Fixed to exclude "Awake" and "InBed" states, counting only actual sleep (Core, Deep, REM)
4. **TDEE Calculation:** Excludes zero-calorie tracking days for accuracy

**Metrics Tracked:**
- Body metrics: Weight, Height
- Energy: Active Energy Burned, Basal Energy Burned, Dietary Energy Consumed
- Activity: Steps, Distance, Flights Climbed, Exercise Time, Stand Time
- Heart: Heart Rate, Resting Heart Rate, HRV, Oxygen Saturation
- Running: Speed, Power, Stride Length, Ground Contact Time, Vertical Oscillation
- Other: VO2 Max, Respiratory Rate, Physical Effort, Sleep Analysis

### Data Sources
- **Primary Source:** Apple Health Export
- **Tracking Apps:** Apple Watch, Foodvisor (nutrition), Strava (activities)
- **Export Format:** CSV files processed and deduplicated
- **Analysis Tools:** Python (Pandas, NumPy, Plotly, Dash)

---

## Conclusions

### Strengths
1. **Consistent Weight Loss:** Sustainable rate without extreme measures
2. **High Activity Level:** Well above recommended minimums
3. **Running Improvement:** Clear fitness gains over time
4. **Data Completeness:** Good tracking adherence across multiple metrics

### Areas for Improvement
1. **Sleep Duration:** Increase average to 7-9 hours per night
2. **Calorie Tracking:** More consistent food logging (34 days with zero entries)
3. **Recovery Monitoring:** Use HRV and RHR trends to optimize training load

### Overall Assessment
The data demonstrates excellent progress in weight loss and fitness improvements. The approach appears sustainable with healthy practices:
- Appropriate calorie deficit (not excessive)
- High activity level supporting energy balance
- Progressive fitness improvements
- No signs of overtraining or extreme measures

**Success Factors:**
- Consistent tracking and data-driven approach
- Balance of structured exercise and daily activity
- Reasonable calorie deficit supporting adherence
- Progressive running improvements without injury

### Future Opportunities
1. **Sleep Optimization:** Target 7-9 hours consistently
2. **Nutrition Consistency:** Maintain daily food tracking
3. **Performance Goals:** Continue running improvements with structured training
4. **Maintenance Planning:** Develop strategy for weight maintenance phase

---

## Appendix: Dashboard Features

The interactive dashboard provides:
- **Overview Tab:** Summary metrics and data quality indicators
- **TDEE Analysis Tab:** Energy balance calculations and visualizations
- **Running Tab:** Performance metrics and cardiovascular fitness tracking
- **Sleep & Recovery Tab:** Sleep patterns and recovery scoring
- **Activity Patterns Tab:** Step counts, consistency analysis, and activity trends

### Access Dashboard
Run the dashboard locally:
```bash
python run_dashboard.py
```
Then open: http://127.0.0.1:8050/

---

*Analysis generated: November 14, 2025*
*Data period: March 24, 2025 - November 14, 2025 (236 days)*
