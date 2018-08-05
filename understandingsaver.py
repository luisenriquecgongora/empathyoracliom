import json
filename = "understanding.json"
while True:
    value = raw_input("Did you understand(Yes:1, No:0)")
    understandingObject = {"understanding":value}
    with open(filename, 'w') as understandingfile:
        json.dump(understandingObject, understandingfile)
