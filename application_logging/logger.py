from datetime import datetime


class AppLogger:
    def __init__(self):
        self.current_time = None
        self.date = None
        self.now = None

    def log(self, file_object, log_message):
        """
        A method that writes logs into a file which is taken as input.
        params: file_object, log_message
        returns: None
        """
        self.now = datetime.now()
        self.date = self.now.date()
        self.current_time = self.now.strftime("%H:%M:%S")
        file_object.write(
            str(self.date) + "/" + str(self.current_time) + "\t\t" + log_message + "\n")
