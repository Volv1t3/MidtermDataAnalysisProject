# Dataset Validation Summary
## Coffee Shop Sales - Maven Roasters Analysis Project

**Date of Analysis:** April 28, 2026  
**Project Course:** ADM-2003 - Análisis de Datos  
**Dataset Source:** Maven Roasters Coffee Shop Sales (Kaggle)

---

## 1. DATASET CHARACTERISTICS - VALIDATION RESULTS

### ✅ Confirmed Dataset Schema

| Characteristic | Expected | Actual | Status |
|---------------|----------|--------|--------|
| **Total Records** | 149,116 transactions | 149,116 transactions | ✅ CONFIRMED |
| **Time Period** | 6 months | January 1 - June 30, 2023 (180 days) | ✅ CONFIRMED |
| **Number of Columns** | Not specified | 11 columns | ✅ VALIDATED |
| **Store Locations** | 3 locations in NYC | 3 locations | ✅ CONFIRMED |
| **Missing Values** | Not specified | 0 missing values | ✅ CLEAN DATASET |

### Dataset Structure

**Columns and Data Types:**
1. `transaction_id` (int64) - Unique transaction identifier
2. `transaction_date` (datetime64) - Date of transaction
3. `transaction_time` (object) - Time of transaction (HH:MM:SS format)
4. `transaction_qty` (int64) - Quantity of items purchased
5. `store_id` (int64) - Store identifier (3, 5, or 8)
6. `store_location` (object) - Store location name
7. `product_id` (int64) - Product identifier
8. `unit_price` (float64) - Price per unit in USD
9. `product_category` (object) - Product category classification
10. `product_type` (object) - Product type classification
11. `product_detail` (object) - Specific product name/variant

---

## 2. KEY FINANCIAL METRICS

### Overall Business Performance

| Metric | Value |
|--------|-------|
| **Total Revenue** | $698,812.33 |
| **Average Transaction Value (Ticket)** | $4.69 |
| **Median Transaction Value** | $3.75 |
| **Total Units Sold** | 214,470 units |
| **Average Units per Transaction** | 1.44 units |
| **Transactions with 1 Item** | 87,159 (58.5%) |
| **Transactions with 2+ Items** | 61,957 (41.5%) |

### Store Performance Comparison

| Store Location | Total Revenue | Transactions | Units Sold | Avg Ticket |
|----------------|---------------|--------------|------------|------------|
| **Hell's Kitchen** | $236,511.17 | 50,735 | 71,737 | $4.66 |
| **Astoria** | $232,243.91 | 50,599 | 70,991 | $4.59 |
| **Lower Manhattan** | $230,057.25 | 47,782 | 71,742 | **$4.81** ⭐ |

**Key Insight:** Lower Manhattan has the highest average ticket ($4.81) despite having fewer transactions, suggesting different customer behavior or product mix.

### Revenue Leaders by Category

| Rank | Product Category | Total Revenue | % of Total | Transactions |
|------|------------------|---------------|------------|--------------|
| 1 | **Coffee** | $269,952.45 | 38.6% | 58,416 |
| 2 | **Tea** | $196,405.95 | 28.1% | 45,449 |
| 3 | **Bakery** | $82,315.64 | 11.8% | 22,796 |
| 4 | **Drinking Chocolate** | $72,416.00 | 10.4% | 11,468 |
| 5 | **Coffee beans** | $40,085.25 | 5.7% | 1,753 |
| 6 | **Branded** | $13,607.00 | 1.9% | 747 |
| 7 | **Loose Tea** | $11,213.60 | 1.6% | 1,210 |
| 8 | **Flavours** | $8,408.80 | 1.2% | 6,790 |
| 9 | **Packaged Chocolate** | $4,407.64 | 0.6% | 487 |

**Key Insight:** Coffee and Tea dominate with 66.7% of total revenue. The top 4 categories account for 88.9% of all revenue.

### Top 10 Products by Revenue

| Rank | Product | Revenue | Transactions |
|------|---------|---------|--------------|
| 1 | Sustainably Grown Organic Lg | $21,151.75 | 2,961 |
| 2 | Dark chocolate Lg | $21,006.00 | 3,029 |
| 3 | Latte Rg | $19,112.25 | 2,896 |
| 4 | Cappuccino Lg | $17,641.75 | 2,772 |
| 5 | Morning Sunrise Chai Lg | $17,384.00 | 2,830 |
| 6 | Latte | $17,257.50 | 2,990 |
| 7 | Jamaican Coffee River Lg | $16,481.25 | 2,911 |
| 8 | Sustainably Grown Organic Rg | $16,233.75 | 2,842 |
| 9 | Cappuccino | $15,997.50 | 2,793 |
| 10 | Brazilian Lg | $15,109.50 | 2,771 |

---

## 3. CATEGORICAL VARIABLES - COMPLETE REFERENCE

### Store Locations (3 unique)
1. **Astoria**
2. **Hell's Kitchen**
3. **Lower Manhattan**

**Store IDs:** 3 (Astoria), 5 (Lower Manhattan), 8 (Hell's Kitchen)

### Product Categories (9 unique)
1. **Bakery** - Pastries, scones, croissants, biscotti
2. **Branded** - Merchandise (mugs, cups, t-shirts)
3. **Coffee** - Espresso drinks, lattes, cappuccinos, brewed coffee
4. **Coffee beans** - Whole bean coffee for retail
5. **Drinking Chocolate** - Hot chocolate beverages
6. **Flavours** - Syrups and flavor additives
7. **Loose Tea** - Loose leaf tea for retail
8. **Packaged Chocolate** - Retail chocolate products
9. **Tea** - Brewed tea beverages (black, green, herbal, chai)

### Product Types (29 unique)
1. Barista Espresso
2. Biscotti
3. Black tea
4. Brewed Black tea
5. Brewed Chai tea
6. Brewed Green tea
7. Brewed herbal tea
8. Chai tea
9. Clothing
10. Drinking Chocolate
11. Drip coffee
12. Espresso Beans
13. Gourmet Beans
14. Gourmet brewed coffee
15. Green beans
16. Green tea
17. Herbal tea
18. Hot chocolate
19. House blend Beans
20. Housewares
21. Organic Beans
22. Organic Chocolate
23. Organic brewed coffee
24. Pastry
25. Premium Beans
26. Premium brewed coffee
27. Regular syrup
28. Scone
29. Sugar free syrup

### Product Details (80 unique items)
**Coffee Products:**
- Brazilian (Lg, Rg, Sm, Organic)
- Cappuccino (Standard, Lg)
- Columbian Medium Roast (Lg, Rg, Sm)
- Espresso Roast, Espresso shot
- Ethiopia (Lg, Rg, Sm, Standard)
- Jamaican Coffee River (Lg, Rg, Sm, Standard)
- Latte (Standard, Rg)
- Our Old Time Diner Blend (Lg, Rg, Sm, Standard)
- Sustainably Grown Organic (Lg, Rg, Standard)
- Ouro Brasileiro shot
- Primo Espresso Roast

**Tea Products:**
- Earl Grey (Lg, Rg, Standard)
- English Breakfast (Lg, Rg, Standard)
- Lemon Grass (Lg, Rg, Standard)
- Morning Sunrise Chai (Lg, Rg, Standard)
- Peppermint (Lg, Rg, Standard)
- Serenity Green Tea (Lg, Rg, Standard)
- Spicy Eye Opener Chai (Lg, Rg, Standard)
- Traditional Blend Chai (Lg, Rg, Standard)

**Bakery Products:**
- Almond Croissant, Chocolate Croissant, Croissant
- Chocolate Chip Biscotti, Ginger Biscotti, Hazelnut Biscotti
- Cranberry Scone, Ginger Scone, Jumbo Savory Scone, Oatmeal Scone, Scottish Cream Scone

**Chocolate Products:**
- Dark chocolate (Lg, Rg, Standard)
- Chili Mayan, Civet Cat, Organic Decaf Blend

**Syrups:**
- Carmel syrup, Chocolate syrup, Hazelnut syrup, Sugar Free Vanilla syrup

**Branded Merchandise:**
- I Need My Bean! Diner mug
- I Need My Bean! Latte cup
- I Need My Bean! T-shirt

**Specialty Items:**
- Guatemalan Sustainably Grown (beans)

---

## 4. TEMPORAL PATTERNS

### Monthly Revenue Trend
| Month | Revenue | Growth vs Previous Month |
|-------|---------|--------------------------|
| January 2023 | $81,677.74 | Baseline |
| February 2023 | $76,145.19 | -6.8% |
| March 2023 | $98,834.68 | +29.8% |
| April 2023 | $118,941.08 | +20.3% |
| May 2023 | $156,727.76 | +31.8% |
| June 2023 | $166,485.88 | +6.2% |

**Key Insight:** Strong upward trend with revenue doubling from January to June. Seasonal pattern suggests increasing demand in spring/summer months.

### Day of Week Performance
| Day | Revenue | Transactions | Avg Ticket |
|-----|---------|--------------|------------|
| Monday | $101,677.28 | 21,643 | $4.70 |
| Tuesday | $99,455.94 | 21,202 | $4.69 |
| Wednesday | $100,313.54 | 21,310 | $4.71 |
| Thursday | $100,767.78 | 21,654 | $4.65 |
| Friday | $101,373.00 | 21,701 | $4.67 |
| Saturday | $96,894.48 | 20,510 | $4.72 |
| Sunday | $98,330.31 | 21,096 | $4.66 |

**Key Insight:** Relatively consistent performance across all days. Weekdays slightly outperform weekends in total revenue but not in average ticket size.

### Peak Hours (Top 5 by Revenue)
1. **10:00 - 11:00** → $88,673.39
2. **09:00 - 10:00** → $85,169.53
3. **08:00 - 09:00** → $82,699.87
4. **07:00 - 08:00** → $63,526.47
5. **11:00 - 12:00** → $46,319.14

**Key Insight:** Morning rush (7am-11am) dominates revenue generation, accounting for approximately 52% of daily revenue.

---

## 5. DISCREPANCIES FROM SYSTEM PROMPT

### No Major Discrepancies Found

The dataset aligns perfectly with the description provided. However, the following details were NOT mentioned in the original description but are now confirmed:

1. **Exact Date Range:** January 1 - June 30, 2023 (not just "6 months")
2. **Total Revenue:** $698,812.33 (not mentioned)
3. **Average Ticket:** $4.69 (not mentioned)
4. **Specific Store Names:** Astoria, Hell's Kitchen, Lower Manhattan (only "three NYC locations" was mentioned)
5. **Number of Unique Products:** 80 distinct product variants
6. **Revenue Growth Pattern:** Strong positive trend from Jan to June (+104% growth)
7. **Time Granularity:** Transaction-level data includes specific timestamps, not just dates

### Data Quality Notes
- ✅ **Zero missing values** - Dataset is 100% complete
- ✅ **Consistent data types** - All fields properly formatted
- ✅ **Logical consistency** - All transactions have valid store IDs, product IDs, positive prices
- ✅ **Temporal consistency** - All dates within stated 6-month period
- ⚠️ **Unit Price Interpretation:** The `unit_price` column represents price per single unit. Total transaction revenue = `unit_price × transaction_qty`

---

## 6. KEY INSIGHTS FROM PROJECT PDF

### Project Context
- **Course:** ADM-2003 - Análisis de Datos (Data Analysis)
- **Institution:** Tecnológico de Monterrey
- **Scenario:** Maven Roasters hired your team as data consultants to analyze their 6-month operational data
- **Core Question:** *"What would you change in this process and how would you support it with the data?"*

### Deliverable Requirements

#### 1. Interactive Dashboard (5 points)
**Must Include:**
- Minimum **5 relevant visualizations**
- **KPI cards** with key business indicators
- **Interactive filters** (by category, date, range, or other relevant variables)
- **Public deployment** (Netlify Drop, CodePen, or professor-approved platform)

**Deliverable:** Public URL link (not the file itself)

#### 2. AI Data Analysis Presentation - PDF (6 points)
**Required Sections:**
1. **Business Context** - Company description, process documented, problem to solve
2. **Analysis Questions** - The 3 guiding questions + 2 additional questions identified by the team
3. **Findings** - Insights supported by dashboard visualizations
4. **Recommendations** - Minimum 3 concrete actions for Maven Roasters' management
5. **AI Process Reflection** - What worked, what needed correction, what AI couldn't do alone

**Guiding Questions from PDF:**
- How do sales behave over time and at different times of day?
- Which products and categories drive the business, and which are underperforming?
- Do the three locations operate the same way or does each follow different logic?

**Design Requirements:**
- White background with high-contrast colors (projector compatibility)
- Avoid dark colors, very light colors, or pastels
- Include APA citations for sources

#### 3. Process Documentation (4 points)
**Must Include:**
- Screenshots of prompts used with AI tools
- Description of iterative evolution of results
- Reflection on learnings: what worked, what didn't, better understanding of AI

**No format requirements** - Focus on authentic evidence of the process

#### 4. Oral Presentation (5 points - individual)
**Requirements:**
- **Date:** Monday, May 11, 2026, 09:00-11:00
- **Location:** Aula D317
- **Duration:** 14 minutes per group
- **Participation:** ALL members must speak and answer questions about any part of the analysis
- **Show live dashboard** during presentation

**Note:** Group project grade is shared, but individual presentation performance is evaluated separately. Poor performance = individual grade penalty without affecting team.

### Evaluation Criteria

| Criterion | Weight | Key Focus |
|-----------|--------|-----------|
| **A. Process Diagnosis & Recommendations** | 6 pts | Understanding the business process, data-supported findings, specific (not generic) recommendations, analytical criterion in additional questions |
| **B. Interactive Dashboard** | 5 pts | Deployed and accessible, 5+ relevant visualizations, well-selected KPIs, functional filters, coherent design |
| **C. AI Process** | 4 pts | Evidence of real iteration, reflection on AI strengths/weaknesses, documented learnings |
| **D. Oral Presentation** | 5 pts | All members participate equally, demonstrate complete analysis knowledge, fluent delivery, concrete Q&A responses |
| **TOTAL** | **20 pts** | **25% of final course grade** |

### Penalties
- Dashboard not deployed or no functional link: **-5 points**
- Team member doesn't speak during presentation: **-5 points (individual)**
- No process document submitted: **-4 points**
- Duration outside range (< 10 min or > 14 min): **-3 points**

### Optional Extra Credit (+1 point to final course grade)
**Quantify One Recommendation:**
- Take one recommendation and calculate the financial impact
- Example: If average ticket increases from $4.69 to $5.50 with same customer volume, what's the monthly and annual revenue increase?
- Format: 1-page appendix in the presentation PDF
- Include: starting number (from dataset), assumption used, calculated result with interpretation

### Submission Details
- **Platform:** D2L
- **Deadline:** Monday, May 11, 2026 at 08:59 (before presentation)
- **File Naming:**
  - Presentation: `AnalisisIA_Apellido1_Apellido2_Apellido3_Apellido4.pdf`
  - Process Doc: `ProcesoIA_Apellido1_Apellido2_Apellido3_Apellido4.pdf`
  - Dashboard: Public URL link

---

## 7. STRATEGIC RECOMMENDATIONS FOR ANALYSIS

Based on the validated dataset and project requirements, the analysis should focus on:

### Priority Analysis Areas

1. **Revenue Optimization**
   - Product mix analysis (high vs low performers)
   - Cross-selling opportunities (41.5% of transactions have 2+ items)
   - Pricing strategy review (wide range from $0.80 to $45.00)

2. **Operational Efficiency**
   - Peak hour staffing optimization (10am peak generates 12.7% of revenue)
   - Store-specific strategies (Lower Manhattan has higher ticket but fewer transactions)
   - Day-part performance (morning vs afternoon vs evening)

3. **Growth Opportunities**
   - Capitalize on upward monthly trend (+104% Jan to June)
   - Weekend revenue boost strategies (slightly underperforming)
   - High-margin category expansion (Coffee beans: $21.02 avg transaction vs $3.38 overall)

4. **Product Portfolio Management**
   - Top 10 products generate significant revenue concentration
   - Long tail analysis (80 products - which should be discontinued?)
   - Category strategy (Flavours: high transaction count but low revenue - why?)

### Dashboard Must-Haves (Based on Data)

**Essential KPIs:**
1. Total Revenue: $698,812.33
2. Average Ticket: $4.69
3. Total Transactions: 149,116
4. Total Units Sold: 214,470
5. Growth Rate: +104% (Jan vs June)

**Essential Visualizations:**
1. Monthly revenue trend line chart (shows growth trajectory)
2. Revenue by product category (bar/pie chart showing Coffee/Tea dominance)
3. Store performance comparison (clustered bar chart)
4. Hourly heatmap (time-of-day patterns)
5. Top 10 products by revenue (horizontal bar chart)
6. Day of week performance (bar chart)
7. Transaction quantity distribution (1 item vs 2+ items)

**Essential Filters:**
- Date range selector (month, custom range)
- Store location dropdown
- Product category selector
- Time of day slider
- Day of week selector

---

## 8. QUESTIONS TO GUIDE ANALYSIS

### Required Questions (From PDF)
1. ✅ How do sales behave over time and at different times of day?
2. ✅ Which products and categories drive the business, and which are underperforming?
3. ✅ Do the three locations operate the same way or follow different logic?

### Suggested Additional Questions (Team Should Define 2)
**Examples:**
- What is the optimal product mix to maximize revenue per transaction?
- How can we increase the percentage of multi-item transactions from 41.5% to 50%?
- Which underperforming products should be discontinued or promoted?
- What pricing strategy could increase the average ticket from $4.69 to $5.50?
- How should staffing be optimized based on hourly revenue patterns?

---

## 9. DATA FILES GENERATED

| File | Location | Purpose |
|------|----------|---------|
| Source Dataset | `/home/ubuntu/Uploads/Coffee Shop Sales.xlsx` | Original data |
| Validation Script | `/home/ubuntu/validate_dataset.py` | Initial schema validation |
| Detailed Analysis Script | `/home/ubuntu/detailed_analysis.py` | Comprehensive statistics |
| Summary JSON | `/home/ubuntu/dataset_summary.json` | Programmatic access to key stats |
| **This Report** | `/home/ubuntu/dataset_validation_summary.md` | Complete validation summary |

---

## 10. NEXT STEPS

### Immediate Actions
1. ✅ Dataset validated - Ready for dashboard development
2. ⏭️ Begin dashboard design and visualization development
3. ⏭️ Formulate 2 additional analytical questions
4. ⏭️ Draft specific business recommendations based on data insights

### Dashboard Development Workflow
1. Design wireframe with 5+ visualizations and KPIs
2. Develop interactive HTML dashboard using AI assistance
3. Test filters and interactivity
4. Deploy to public URL (Netlify Drop recommended)
5. Verify accessibility and responsiveness

### Presentation Development Workflow
1. Create presentation outline covering all required sections
2. Generate visualizations from dashboard for findings section
3. Develop 3+ specific, actionable recommendations for Maven Roasters
4. Write AI process reflection with authentic insights
5. Format with white background and high-contrast colors
6. Add APA citations

### Process Documentation Workflow
1. Screenshot all AI prompts used throughout
2. Document iteration cycles (what changed and why)
3. Reflect on AI strengths and limitations discovered
4. Compile into process document

---

## APPENDIX: Quick Reference Statistics

**Dataset Snapshot:**
- 📊 **149,116** transactions
- 📅 **180 days** (Jan 1 - Jun 30, 2023)
- 💰 **$698,812** total revenue
- 🎫 **$4.69** average ticket
- 🏪 **3** store locations
- 📦 **9** product categories
- 🎯 **80** unique products

**Top Revenue Generators:**
1. Coffee: $269,952 (38.6%)
2. Tea: $196,406 (28.1%)
3. Bakery: $82,316 (11.8%)

**Best Performing:**
- 📍 Store: Hell's Kitchen ($236,511)
- 📈 Month: June 2023 ($166,486)
- ⏰ Hour: 10:00-11:00 AM ($88,673)
- ☕ Product: Sustainably Grown Organic Lg ($21,152)

---

**Report Generated:** April 28, 2026  
**Status:** ✅ Dataset validated and ready for analysis  
**Confidence Level:** HIGH - Zero data quality issues detected
