import json


class DbManager:
    def __init__(self, pathToDb) -> None:
        self.pathToFile = pathToDb

    def loadFile(self, fileName) -> dict:
        with open(self.pathToFile + "/" + fileName, "r") as f:
            data = json.load(f)

        return data

    def writeFile(self, fileName, data) -> None:
        with open(self.pathToFile + "/" + fileName, "w") as f:
            json.dump(data, f)
