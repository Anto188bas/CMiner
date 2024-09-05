def read_records(filename):
    with open(filename, 'r') as file:
        content = file.read()
    # Split the content into records
    records = content.split('------------------------------------------')
    # Clean up records by removing leading/trailing whitespace and empty strings
    records = [record.strip() for record in records if record.strip()]
    return records

def find_unique_records(file1_records, file2_records):
    unique_records = [record for record in file1_records if record not in file2_records]
    return unique_records

# Read records from both files
file1_records = read_records('1.txt')
file2_records = read_records('2.txt')

# Find unique records in file1
unique_records = find_unique_records(file1_records, file2_records)

print(len(unique_records))
# Print the unique records
for record in unique_records:
    print(f"Unique record in file1:\n{record}\n------------------------------------------")
