import os

class DirectoryHelper:
    @staticmethod
    def root_path():
        """
        Get the root path of the project.
        Returns:
            str
        """

        return os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))