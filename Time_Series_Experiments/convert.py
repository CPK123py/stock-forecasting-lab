import csv

# Input and output file names
input_file = "zomato_news.csv"  # Replace with your actual input CSV filename
output_file = "zom_news.csv"

# Read the input CSV and process the data
with open(input_file, mode="r", encoding="utf-8") as infile, open(output_file, mode="w", encoding="utf-8", newline="") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Write the new header
    writer.writerow(["date", "news"])

    # Process rows
    data = []
    current_date = None

    for row in reader:
        if len(row) == 1:  # If it's a single-column row (date or news)
            text = row[0].strip()
            if "Read More" not in text:  # Ignore "Read More"
                if "," in text:  # Assuming dates have commas (e.g., "9 March, 2025")
                    current_date = text  # Store the date
                elif current_date:  # If it's news, pair it with the last date
                    writer.writerow([current_date, text])

print(f"✅ Cleaned CSV saved as {output_file}")
