The cvs file Commercial_solid_waste_audit_Belize_data and Household_solid_waste_audit_Belize_data report data about abundance and composition of the solid waste produced by the commercial activitites and the households audited in Belize during the Commonwealth Litter Project (CLiP). The legend for the columns in the file is as follows: 

ID: unique number of the audited commercial activity, house or street litter bag (please note: the ID are unique when only one Premises_type is considered; however, some of the IDs are repeated in both pt_commercial and pt_household)
Premises_type: whether the data is collected from commercial activities(pt_commercial), households (pt_household) or from a bag containing street litter (pt_litter)	
Date: when audit was conducted	
Item: type of item counted in the audit	
Material: material-category of the Item	
Measure: whether the value represent count (number of items), weight (grems) or volume (liters)
Site: Name of the site audited 	
Type: for commercial activity, type of activity; for households, whether they represent urban, rural or regional areas	
Collection frequency: number of days between collection events (from interviews)
Collection efficiency: amount of total waste produced that is collected (from interviews)
Value: original value recorded during the audit	
Correted value: rate of production measured dividing the Value by (Collection frequency * Collection efficiency)


The cvs files Commercial_solid_waste_audit_Belize_metadata and Household_solid_waste_audit_Belize_metadata report metadata collected from households and commercial activities through a questionnaire at the moment of or just after the solid waste sample collection. The legend for the columns in the files is as follows: 

For Commercial_solid_waste_audit_Belize_metadata
ID: the unique premises ID
Date: the date the interview data was submitted
Premises_type: pt_commercial only	
Location: The premises location
Business_Type: type of business Admin, Food_outlet, Retail_shop, Hotels, Supermarket, Other	
Other notes: notes about the premises
Location_lat: WGS84 latitude coordinate
Location_long: WGS84 longitude coordinate
Who_collects_the_rubbish: whether the collection service is run privately or by the council	
Collection_period: how often waste is collected; categorical. p_xx means the waste is collected every xx period (see accumulation_time_num) – householder reported (categorical)	
Accumulation_time_num: numerical version of collection_period	
Collection_rating: collection satisfaction on a 1-5 scale	
Waste_disposal_green: method of disposal of green waste (categorical: collect, burn, dump on land, dump in water, bury, other)
Waste_disposal_general: method of disposal of general waste (categorical: collect, burn, dump on land, dump in water, bury, other)
Comments: any additional comment	
Collected: fraction of waste collected as recorded at pickup location	
Number_of_Employees: number of people employed by the business 
Floor_Space: area covered by the business	
Floor_Space_Units: unit of measure of the floor space	
Number_of_Rooms: number of rooms in the premises


For Household_solid_waste_audit_Belize_metadata
ID: the unique premises ID
Date: the date the interview data was submitted
Premises_type: pt_household for households or pt_litter for bags of street litter
Location: the premises location
Urbanity: level of urbanity
Location_lat: WGS84 latitude coordinate
Location_long: WGS84 longitude coordinate
Currency: currency of all responses involving money in this row
Bins_number: how many bins were found at the house
Bins_size_material: size and material of garbage bins (plaintext)
Collected: fraction of waste collected as recorded at pickup location
Collection_available: is garbage collection available for this house?
Collection_info: has the household received information about garbage collection?
Collection_info_channel: how has collection information been received? (plaintext)
Collection_info_channel_interp: how has collection information been received? (categorical: council, enquiring in person, collection worker, social media, mail, observation, family, neighbours, school, tv, village meeting)
Collection_period: how often waste is collected; categorical. p_xx means the waste is collected every xx period (see accumulation_time_num) – householder reported (categorical)
Accumulation_time_num: numerical version of collection_period
Collection_period_other: how often waste is collected – householder reported (plaintext)
Collection_rating: collection satisfaction on a 1-5 scale
Collection_rating_reason: reason for collection rating (plaintext)
Collection_waste_level_when_ta: waste level when it is collected (categorical: empty, half_full, full, overflowing)
Collection_rubbish_bag_support: response to question “Do you support an idea of introducing a prepaid rubbish bag for people to put their waste in? When you purchase a specific bag the price includes the price for the waste to be collected as well. This means if you produce less waste, you pay less.”
Collection_rubbish_bag_willing: response to question “Assume that to dispose of rubbish you must buy a bag, how much would you be willing to pay for a single bag.”
Bow_n_bow_collection_opinion: response to question “Would you like to see the Bowen and Bowen return system extended to other products like shampoo bottles, old TVs, Fans, and all other cans? Bowen and Bowen already accepts returns of water and beer bottles for a monetary exchange.”
Collection_suggestions: suggestions for improving waste collection
Collection_willing_to_pay: response to “How much are you willing to pay for waste collection” (no period recorded)
Waste_disposal_bulky: method of disposal of bulky waste (categorical: collect, burn, dump on land, dump in water, bury, other)
Waste_disposal_bulky_other: method of disposal of bulky waste – “other” option selected
Waste_disposal_general: method of disposal of general waste (categorical: collect, burn, dump on land, dump in water, bury, other)
Waste_disposal_general_other: method of disposal of general waste – “other” option selected
Waste_disposal_green: method of disposal of green waste (categorical: collect, burn, dump on land, dump in water, bury, other)
Waste_disposal_green_other: method of disposal of green waste – “other” option selected
Waste_disposal_nappies: method of disposal of nappies waste (categorical: collect, burn, dump on land, dump in water, bury, other)
Waste_disposal_nappies_other: method of disposal of nappies waste – “other” option selected
Diet_food #: most commonly eaten food rank # (1-3)
Diet_frequency #: how often rank # food is eaten
Diet_source #: how rank # food is obtained (buy or produce)
Grocery_expenses: zmount reported spent on groceries
Grocery_period: how often money is spent on groceries (categorical)
Grocery_period_num: numerical version of grocery_period
Grocery_period_other: how often money is spent on groceries (if “other” was chosen for the period)
Grocery_total: Total monthly spend on groceries
House_number_people: number of people in the house
House_number_children: number of children in the house
House_number_people_income: number of people with income source in house
House_ownership: categorical - own, rent
What_sources_of_income_does_this_person_have 0: income source for earner #
Income_amount #: reported income of the # income earner in the house
Income_period #: how often income earner # is paid (categorical)
Income_period_num #: Numerical version of income period
Income_period_other #: Specification of income period if interviewee said 'other'
Income_total: Total monthly income for all earners in house
Levy_support: Response to question “In order for cans, plastic bottles, and bulky waste to be recycled, we need to support the cost by introducing a waste levy like other countries. Do you support this?“
Electricity_expenses_period: Electricity expenses over given period (plaintext)
Mobile_phone: Does the house have a mobile phone?
Softdrink_cans: Reported soft drink cans consumed per week
Transportation_expenses_period: Transport expenses over given period (plaintext)
Waterbottles_p_week: Reported number of water bottles consumed per week
Total_containers: Count, number of items
Cigarette_butts: Count, number of items
Cigarette_packets: Count, number of items
Straws: Count, number of items
Coffee_cups: Count, number of items
Bags_glossy: Count, number of items
Bags_supermarket: Count, number of items
Takeaway_conts: Count, number of items
Takeaway_lids: Count, number of items

The data were quality checked at by Asia-Pacific Waste Consultants and by Cefas operators checking for discrepancies between paper and digital data and for other mistakes (presence of commas to avoid errors in csv conversion, missing data).