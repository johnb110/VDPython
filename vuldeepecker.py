"""
Interface to VulDeePecker project
"""
import sys
import pandas
from CleanGadget import clean_gadget
from tokenize_gadget import tokenize_df

def vectorize(gadget):
    return gadget

class NeuralNet:
    def __init__(self, data):
        self.data = data
    
    def train(self):
        pass
    
    def test(self):
        print("Accuracy is...")
        pass

"""
Parses gadget file to find individual gadgets
Yields each gadget as list of strings, where each element is code line
Has to ignore first line of each gadget, which starts as integer+space
At the end of each code gadget is binary value
    This indicates whether or not there is vulnerability in that gadget
"""
def parse_file(filename):
    with open(filename, "r", encoding="utf8") as file:
        gadget = []
        gadget_val = 0
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            if "-" * 33 in line and gadget: 
                yield clean_gadget(gadget), gadget_val
                gadget = []
            elif stripped.split()[0].isdigit():
                if gadget:
                    # Code line could start with number (somehow)
                    if stripped.isdigit():
                        gadget_val = int(stripped)
                    else:
                        gadget.append(stripped)
            else:
                gadget.append(stripped)
            
"""
Use parse_file to get gadgets
Use vectorize function to get gadget vectors
Assuming all gadgets can fit in memory, build list of gadget dictionaries
    Dictionary contains gadget vector and vulnerability indicator
Convert list of dictionaries to dataframe when all gadgets are processed
Instantiate neural network, pass data to it, train, test, print accuracy
"""
def main():
    if len(sys.argv) != 2:
        print("Usage: python vuldeepecker.py [filename]")
        exit()
    filename = sys.argv[1]
    parse_file(filename)
    gadgets = []
    count = 0
    for gadget, val in parse_file(filename):
        count += 1
        print("Collecting gadgets...", count, end="\r")
        vector = vectorize(gadget)
        row = {"gadget" : vector, "val" : val}
        gadgets.append(row)
    df = pandas.DataFrame(gadgets)
    df = tokenize_df(df)

    print(df['tokenized_gadget'][0])
    nn = NeuralNet(df)
    nn.train()
    nn.test()

if __name__ == "__main__":
    main()