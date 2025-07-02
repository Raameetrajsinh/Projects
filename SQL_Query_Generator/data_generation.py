import sqlite3

conn = sqlite3.connect("students.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS students (
    StudentID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT NOT NULL,
    DateOfBirth DATE NOT NULL,
    Gender TEXT CHECK(Gender IN ('Male', 'Female')) DEFAULT 'Female',
    EnrollmentDate DATE NOT NULL DEFAULT (date('now')),
    Grade INTEGER
)
""")

sample_students = [
    ("Alice Johnson", "2002-05-14", "Female", "2020-09-01", 89),
    ("Bob Smith", "2001-11-30", "Male", "2019-09-01", 76),
    ("Clara Lopez", "2003-03-22", "Female", "2021-01-15", 92),
    ("Daniel Wu", "2000-07-10", "Male", "2018-09-01", 68),
    ("Eva Martins", "2004-01-05", "Female", "2022-01-10", 81),
    ("Frank Brown", "2002-09-18", "Male", "2020-09-01", 85),
    ("Grace Green", "2001-04-02", "Female", "2019-09-01", 79),
    ("Henry White", "2003-12-12", "Male", "2021-01-15", 91),
    ("Isabel Black", "2000-06-21", "Female", "2018-09-01", 74),
    ("Jack Clark", "2004-02-14", "Male", "2022-01-10", 88),
    ("Kara Wilson", "2002-11-30", "Female", "2020-09-01", 90),
    ("Liam Hall", "2001-07-07", "Male", "2019-09-01", 77),
    ("Mia Young", "2003-05-19", "Female", "2021-01-15", 93),
    ("Noah King", "2000-03-23", "Male", "2018-09-01", 69),
    ("Olivia Wright", "2004-08-28", "Female", "2022-01-10", 82),
    ("Paul Scott", "2002-01-11", "Male", "2020-09-01", 84),
    ("Quinn Adams", "2001-10-30", "Female", "2019-09-01", 78),
    ("Rachel Baker", "2003-04-04", "Female", "2021-01-15", 90),
    ("Steve Carter", "2000-12-15", "Male", "2018-09-01", 66),
    ("Tina Davis", "2004-09-09", "Female", "2022-01-10", 85),
]

cursor.executemany("""
INSERT INTO students (Name, DateOfBirth, Gender, EnrollmentDate, Grade)
VALUES (?, ?, ?, ?, ?)
""", sample_students)

conn.commit()
conn.close()

print("✅ students.db created with 20 fixed student rows.")