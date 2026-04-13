import random
from supabase_config import supabase

def generate_and_insert_data(num_records=1000):
    response = supabase.table('students_dataset').select('id', count='exact').limit(1).execute()
    count = response.count if response.count is not None else 0
    if count >= num_records:
        print(f"Dataset already contains {count} records. Skipping generation.")
        return

    print(f"Generating {num_records} synthetic records using Supabase API...")
    
    # We can perform batch inserts in chunks of 500
    chunk_size = 500
    for i in range(0, num_records, chunk_size):
        records = []
        for _ in range(chunk_size):
            attendance = random.randint(30, 100)
            study_hours = round(random.uniform(0.5, 10.0), 1)
            internal_marks = random.randint(5, 50)
            assignments = random.randint(5, 50)
            previous_gpa = round(random.uniform(3.0, 10.0), 2)
            
            score = (
                (attendance / 100.0) * 0.2 + 
                (study_hours / 10.0) * 0.2 + 
                (internal_marks / 50.0) * 0.2 + 
                (assignments / 50.0) * 0.2 + 
                (previous_gpa / 10.0) * 0.2
            )
            score += random.uniform(-0.1, 0.1)
            
            result = 1 if score > 0.55 else 0
            
            records.append({
                "attendance": attendance,
                "study_hours": study_hours,
                "internal_marks": internal_marks,
                "assignments": assignments,
                "previous_gpa": previous_gpa,
                "result": result
            })
            
        supabase.table('students_dataset').insert(records).execute()
        print(f"Inserted chunk of {chunk_size} records.")

    print("Data inserted successfully.")

if __name__ == "__main__":
    generate_and_insert_data(1000)
