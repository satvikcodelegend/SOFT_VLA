import csv
import os

class DataLogger:

    def __init__(self, filename="simulation_log.csv"):

        os.makedirs("data", exist_ok=True)

        self.filepath = os.path.join("data", filename)

        self.file = open(self.filepath, "w", newline="")

        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "time",

            "tip_x",
            "tip_y",
            "tip_z",

            "target_x",
            "target_y",
            "target_z",

            "command_p1",
            "command_p2",
            "command_p3",

            "filtered_p1",
            "filtered_p2",
            "filtered_p3"
        ])


    def log(self,
            time,
            tip,
            target,
            command_pressure,
            filtered_pressure):

        self.writer.writerow([

            time,

            tip[0],
            tip[1],
            tip[2],

            target[0],
            target[1],
            target[2],

            command_pressure[0],
            command_pressure[1],
            command_pressure[2],

            filtered_pressure[0],
            filtered_pressure[1],
            filtered_pressure[2]

        ])


    def close(self):
        self.file.close()
    