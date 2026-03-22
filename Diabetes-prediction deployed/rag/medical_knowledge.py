# DiabetesAI Medical Knowledge Base
# Sources: WHO Guidelines, ADA Standards, NIH Recommendations

MEDICAL_DOCUMENTS = [

    # ════════════════════════════════════════
    # DOCUMENT 1: WHO DIABETES GUIDELINES
    # ════════════════════════════════════════
    {
        "id": "who_diagnosis",
        "source": "WHO Diabetes Guidelines 2023",
        "category": "diagnosis",
        "content": """
        Diabetes mellitus is diagnosed when fasting plasma glucose is 7.0 mmol/L (126 mg/dL) or above.
        Pre-diabetes is diagnosed when fasting glucose is between 6.1-6.9 mmol/L (110-125 mg/dL).
        Normal fasting glucose is below 6.1 mmol/L (110 mg/dL).
        A 2-hour plasma glucose of 11.1 mmol/L (200 mg/dL) or above during oral glucose tolerance test confirms diabetes.
        HbA1c of 48 mmol/mol (6.5%) or above is also diagnostic of diabetes.
        Symptoms of diabetes include polyuria (frequent urination), polydipsia (excessive thirst), unexplained weight loss, and blurred vision.
        Type 2 diabetes accounts for 90-95% of all diabetes cases worldwide.
        Risk factors include obesity, physical inactivity, family history, age above 45, and history of gestational diabetes.
        """
    },

    {
        "id": "who_complications",
        "source": "WHO Diabetes Complications Report",
        "category": "complications",
        "content": """
        Uncontrolled diabetes leads to serious complications affecting multiple organs.
        Diabetic nephropathy affects 20-40% of people with diabetes and is the leading cause of kidney failure.
        Diabetic retinopathy affects over 75% of people who have had diabetes for 20 years or more.
        Diabetic neuropathy causes numbness, tingling, and pain especially in the feet and hands.
        Cardiovascular disease is 2-3 times more common in people with diabetes.
        Diabetic foot complications can lead to amputation if untreated — inspect feet daily.
        Warning signs requiring immediate medical attention: chest pain, sudden vision loss, numbness in feet, non-healing wounds, very high or very low blood sugar.
        Regular screening for complications should include annual eye exam, kidney function tests, foot examination, and blood pressure monitoring.
        """
    },

    # ════════════════════════════════════════
    # DOCUMENT 2: ADA DIETARY GUIDELINES
    # ════════════════════════════════════════
    {
        "id": "ada_diet_general",
        "source": "American Diabetes Association Standards of Care 2024",
        "category": "diet",
        "content": """
        The ADA recommends a personalised eating plan developed with a registered dietitian for diabetes management.
        There is no single ideal dietary pattern for all people with diabetes.
        Reducing overall carbohydrate intake has the most evidence for improving blood glucose levels.
        Eating patterns that have been shown to be effective include Mediterranean diet, DASH diet, and low-carbohydrate diets.
        Focus on nutrient-dense, high-fiber foods including vegetables, legumes, fruits, whole grains, and lean proteins.
        Minimise intake of processed foods, sugary beverages, refined grains, and high-sodium foods.
        Portion control is essential — use the plate method: half vegetables, quarter lean protein, quarter whole grains.
        Eating at regular times and avoiding skipping meals helps stabilise blood glucose levels throughout the day.
        """
    },

    {
        "id": "ada_foods_avoid",
        "source": "ADA Nutrition Therapy Recommendations 2024",
        "category": "diet",
        "content": """
        Foods diabetics should avoid or strictly limit:
        Sugary beverages including regular soda, fruit juices, sweet tea, and energy drinks cause rapid blood sugar spikes.
        White bread, white rice, and white pasta are refined carbohydrates with high glycaemic index.
        Fried foods including french fries, fried chicken, and donuts are high in unhealthy fats.
        Full-fat dairy products including whole milk, butter, and cheese should be consumed in moderation.
        Processed meats like sausage, hot dogs, and bacon are associated with increased diabetes complications.
        Packaged snacks like chips, crackers, and cookies often contain refined carbs and trans fats.
        Alcohol should be limited — it can cause hypoglycaemia especially when taken with insulin or some diabetes medications.
        High-sodium foods can raise blood pressure which is already a risk factor in diabetes.
        Sweets, cakes, pastries, and desserts should be occasional treats only, not regular consumption.
        """
    },

    {
        "id": "ada_foods_recommended",
        "source": "ADA Recommended Foods for Diabetes 2024",
        "category": "diet",
        "content": """
        Best foods for people with diabetes:
        Non-starchy vegetables like broccoli, spinach, cauliflower, peppers, and green beans are excellent choices with minimal impact on blood sugar.
        Fatty fish like salmon, mackerel, sardines, and tuna are rich in omega-3 fatty acids which reduce cardiovascular risk.
        Whole grains including oats, brown rice, quinoa, and barley have lower glycaemic index than refined grains.
        Legumes including lentils, chickpeas, black beans, and kidney beans are high in protein and fiber.
        Nuts and seeds including almonds, walnuts, chia seeds, and flaxseeds provide healthy fats and fiber.
        Greek yogurt is a good source of protein and probiotics with lower carbohydrates than regular yogurt.
        Berries including blueberries, strawberries, and raspberries are relatively low in sugar and high in antioxidants.
        Avocado provides healthy monounsaturated fats and helps with blood sugar regulation.
        Eggs are an excellent source of protein and do not significantly affect blood glucose.
        Olive oil is the preferred cooking oil for people with diabetes due to its anti-inflammatory properties.
        """
    },

    {
        "id": "ada_meal_planning",
        "source": "ADA Meal Planning Guide 2024",
        "category": "diet",
        "content": """
        The diabetes plate method: fill half the plate with non-starchy vegetables, quarter with lean protein, quarter with complex carbohydrates.
        Carbohydrate counting is an effective meal planning strategy — most adults with diabetes should aim for 45-60 grams of carbs per meal.
        The glycaemic index measures how quickly foods raise blood sugar — choose low GI foods (below 55) when possible.
        Eating smaller, frequent meals every 3-4 hours helps maintain stable blood glucose levels.
        Never skip breakfast — it leads to blood sugar irregularities throughout the day.
        Drink water as the primary beverage — aim for 8 glasses per day.
        Cooking methods matter — baking, grilling, steaming, and boiling are healthier than frying.
        Reading food labels helps identify hidden sugars — look for words ending in -ose, syrup, and concentrate.
        Indian foods suitable for diabetics: dal, sabzi, raita with cucumber, sprouts, bitter gourd (karela), fenugreek (methi).
        """
    },

    # ════════════════════════════════════════
    # DOCUMENT 3: EXERCISE GUIDELINES
    # ════════════════════════════════════════
    {
        "id": "exercise_recommendations",
        "source": "ADA Physical Activity Standards 2024",
        "category": "exercise",
        "content": """
        Physical activity is a cornerstone of diabetes management and prevention.
        Adults with diabetes should aim for at least 150 minutes per week of moderate-intensity aerobic activity.
        Aerobic exercises suitable for diabetics include brisk walking, swimming, cycling, dancing, and water aerobics.
        Resistance training at least 2-3 times per week helps improve insulin sensitivity.
        Reducing sedentary time by breaking up sitting with short walks every 30 minutes improves glucose regulation.
        Exercise lowers blood glucose by increasing insulin sensitivity for up to 24-48 hours after activity.
        Check blood sugar before and after exercise — exercise can cause hypoglycaemia in people on insulin.
        Start slowly and gradually increase intensity — even 10-minute walks 3 times daily provide benefits.
        Morning exercise before breakfast helps control blood sugar throughout the day.
        Yoga and meditation reduce cortisol levels which directly impacts blood sugar regulation.
        """
    },

    # ════════════════════════════════════════
    # DOCUMENT 4: MEDICATIONS AND MONITORING
    # ════════════════════════════════════════
    {
        "id": "medications_overview",
        "source": "ADA Pharmacological Treatment Guidelines 2024",
        "category": "medication",
        "content": """
        Metformin is the first-line medication for Type 2 diabetes — it reduces liver glucose production and improves insulin sensitivity.
        SGLT2 inhibitors like empagliflozin and dapagliflozin reduce blood sugar and have heart and kidney protective effects.
        GLP-1 receptor agonists like semaglutide and liraglutide lower blood sugar and promote weight loss.
        DPP-4 inhibitors like sitagliptin are well-tolerated with low risk of hypoglycaemia.
        Insulin therapy may be required for Type 2 diabetes when oral medications are insufficient.
        Sulfonylureas stimulate the pancreas to produce more insulin but can cause hypoglycaemia.
        Blood pressure medications like ACE inhibitors are often prescribed alongside diabetes medications to protect kidneys.
        Statins for cholesterol management are recommended for most adults with diabetes due to cardiovascular risk.
        Never stop or change diabetes medications without consulting your doctor.
        Always carry fast-acting glucose (glucose tablets, juice) in case of hypoglycaemia.
        """
    },

    {
        "id": "blood_sugar_monitoring",
        "source": "ADA Blood Glucose Monitoring Guidelines 2024",
        "category": "monitoring",
        "content": """
        Target blood glucose levels for most adults with diabetes: fasting 80-130 mg/dL, 2 hours after meals below 180 mg/dL.
        HbA1c target for most adults with diabetes is below 7% (53 mmol/mol).
        Self-monitoring of blood glucose should be done as recommended by your healthcare provider.
        Continuous Glucose Monitors (CGM) provide real-time glucose readings and trend information.
        Signs of hypoglycaemia (low blood sugar below 70 mg/dL): shakiness, sweating, confusion, irritability, rapid heartbeat.
        Treat hypoglycaemia immediately with 15 grams of fast-acting carbohydrates — 4 glucose tablets, half cup of juice, or 3 teaspoons of sugar.
        Signs of hyperglycaemia (high blood sugar): increased thirst, frequent urination, fatigue, blurred vision, headache.
        Diabetic ketoacidosis (DKA) is a medical emergency — symptoms include nausea, vomiting, fruity-smelling breath, confusion.
        Keep a blood sugar diary to identify patterns and share with your doctor at appointments.
        """
    },

    # ════════════════════════════════════════
    # DOCUMENT 5: LIFESTYLE AND PREVENTION
    # ════════════════════════════════════════
    {
        "id": "lifestyle_management",
        "source": "WHO Diabetes Prevention Programme 2023",
        "category": "lifestyle",
        "content": """
        Lifestyle modification can reduce the risk of developing Type 2 diabetes by 58% in high-risk individuals.
        Weight loss of 5-10% of body weight significantly improves blood sugar control.
        Sleep quality directly impacts blood sugar — aim for 7-9 hours of quality sleep per night.
        Chronic stress raises cortisol which increases blood glucose — practice stress management techniques.
        Stress management techniques: deep breathing, progressive muscle relaxation, meditation, yoga, and regular social activities.
        Smoking cessation is critical — smoking significantly increases cardiovascular risk in people with diabetes.
        Limit alcohol to no more than 1 drink per day for women, 2 for men — always with food to prevent hypoglycaemia.
        Regular dental checkups are important — gum disease is more common and severe in people with diabetes.
        Foot care: wash and dry feet daily, moisturise, check for cuts or sores, wear comfortable shoes, never go barefoot.
        Annual flu vaccine and pneumococcal vaccine are recommended for people with diabetes.
        """
    },

    {
        "id": "prevention_strategies",
        "source": "ADA Prevention of Type 2 Diabetes 2024",
        "category": "prevention",
        "content": """
        Pre-diabetes is reversible — making lifestyle changes can prevent or delay progression to Type 2 diabetes.
        The Diabetes Prevention Program showed that intensive lifestyle changes reduced diabetes risk by 58% over 3 years.
        Key prevention strategies: achieve and maintain healthy weight, increase physical activity, improve diet quality.
        People with pre-diabetes should be screened for diabetes every 1-2 years.
        Moderate weight loss of 7% of body weight combined with 150 minutes of weekly exercise is the most effective prevention.
        High-risk groups should be screened: people over 45, those with family history, overweight individuals, people of South Asian descent.
        Metformin can be prescribed for diabetes prevention in high-risk individuals who cannot achieve lifestyle goals.
        Regular monitoring of blood pressure, cholesterol, and kidney function helps detect early complications.
        Building healthy habits gradually is more sustainable than drastic changes — start with small, achievable goals.
        Community support, family involvement, and regular medical follow-ups improve long-term outcomes.
        """
    },

    # ════════════════════════════════════════
    # DOCUMENT 6: MENTAL HEALTH
    # ════════════════════════════════════════
    {
        "id": "mental_health_diabetes",
        "source": "ADA Mental Health Standards 2024",
        "category": "mental_health",
        "content": """
        Diabetes distress affects 45% of people with diabetes — it is the emotional burden of managing a chronic condition.
        Depression is 2-3 times more common in people with diabetes than in the general population.
        Anxiety about blood sugar levels, complications, and medication management is very common in diabetes.
        Mental health support is an essential part of comprehensive diabetes care.
        Signs of diabetes burnout: neglecting blood sugar checks, skipping medications, poor dietary choices, feeling overwhelmed.
        Seeking help is a sign of strength — talk to your doctor about mental health concerns.
        Cognitive behavioural therapy (CBT) is effective for depression and anxiety in people with diabetes.
        Peer support groups for people with diabetes provide emotional support and practical advice.
        Family and social support significantly improves diabetes management outcomes.
        Remember: having diabetes does not define you — millions of people with diabetes live full, healthy, active lives.
        """
    },

    # ════════════════════════════════════════
    # DOCUMENT 7: SPECIFIC SCENARIOS
    # ════════════════════════════════════════
    {
        "id": "ice_cream_sweets",
        "source": "ADA Practical Guide to Managing Sweet Cravings",
        "category": "diet",
        "content": """
        Sweet cravings are normal and can be managed without complete elimination of all sweets.
        Small portions of regular ice cream occasionally (1/2 cup) can fit into a diabetes meal plan.
        Better alternatives to regular ice cream: frozen banana nice cream, low-sugar frozen yogurt, berries with whipped cream.
        Dark chocolate (70% or higher cocoa) in small amounts (1-2 squares) has lower sugar and may benefit heart health.
        Satisfy sweet cravings with naturally sweet foods: fresh fruits, dates in small quantities, cinnamon on oatmeal.
        Sugar-free alternatives use artificial sweeteners — while safe, they can maintain sweet cravings over time.
        If consuming sweets, pair them with protein or fat to slow glucose absorption — e.g., apple with peanut butter.
        Time sweet treats after physical activity when muscles are more insulin-sensitive.
        Plan for treats in advance rather than impulsive eating to maintain portion control.
        Completely banning favourite foods often leads to binge eating — moderation is more sustainable.
        """
    },

    {
        "id": "eating_out_restaurants",
        "source": "ADA Dining Out Guide for Diabetes",
        "category": "diet",
        "content": """
        Eating out with diabetes requires planning but should not be avoided entirely.
        Best restaurant choices: grilled fish or chicken, salads with dressing on the side, steamed vegetables.
        Ask for sauces, gravies, and dressings on the side to control portions.
        Avoid breaded and fried items — opt for grilled, baked, or steamed preparations.
        Share large portions or ask for a half portion to control carbohydrate intake.
        Indian restaurant tips: choose dal, tandoori items, raita, sabzi; avoid naan, biryani, and sweet lassi.
        Request substitutions: swap fries for salad, white rice for vegetables, regular bread for whole grain.
        Check blood sugar 2 hours after eating out to understand how restaurant meals affect your glucose.
        Inform restaurant staff about diabetes if needed — most establishments can accommodate dietary needs.
        Don't skip meals before eating out to compensate — this leads to overeating and blood sugar swings.
        """
    },

    {
        "id": "bmi_weight_management",
        "source": "WHO BMI and Diabetes Risk Management 2023",
        "category": "lifestyle",
        "content": """
        BMI (Body Mass Index) is calculated as weight in kg divided by height in metres squared.
        Normal BMI range is 18.5-24.9. Overweight is 25-29.9. Obese is 30 and above.
        For South Asian populations, diabetes risk increases at BMI above 23 — lower than Western populations.
        Every 1 kg/m2 increase in BMI above 25 increases diabetes risk by approximately 10-20%.
        Losing 5-10% of body weight reduces HbA1c by 0.5-1.5% in people with Type 2 diabetes.
        Waist circumference is also important — above 80cm for women and 94cm for men indicates increased risk.
        Weight loss methods that work for diabetes: calorie restriction, low-carbohydrate diet, Mediterranean diet.
        Bariatric surgery is effective for diabetes remission in people with BMI above 35 — discuss with your doctor.
        Set realistic weight loss goals — 0.5-1 kg per week is safe and sustainable.
        Combining dietary changes with regular exercise produces better weight loss than either alone.
        """
    }
]