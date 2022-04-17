"""
Stream lit GUI for hosting PyGames
"""

# Imports
import os
import streamlit as st
import json

from main import *

# Main Vars
PATH_EVALPLOT = "eval_plot.png"
LOGS = []
PrintWidget = None

# Main Classes
class NLP_ARGS:
    def __init__(self, dataset, out_folder, segmenter, tokenizer, custom, printObj=print):
        self.dataset = dataset
        self.out_folder = out_folder
        self.segmenter = segmenter
        self.tokenizer = tokenizer
        self.custom = custom
        self.printObj = printObj

# Main Functions
def printObj(text):
        global LOGS
        global PrintWidget
        LOGS.append(str(text))
        PrintWidget.text_area("Logs", "\n".join(LOGS), height=500)

def main():
    global LOGS
    global PrintWidget
    # Titles
    st.title("NLP Project")
    st.markdown("## Group 10")
    
    # Inputs
    dataset = st.text_input("Dataset Path", "cranfield/")
    out_folder = st.text_input("Output Folder Path", "output/")
    segmenter = st.selectbox("Choose Segmenter", ["naive", "punkt"])
    tokenizer = st.selectbox("Choose Tokenizer", ["naive", "ptb"])
    custom = st.checkbox("Custom Query?")
    if custom:
        query = st.text_input("")
    
    # Print Obj
    if st.button("Run"):
        LOGS = []
        
        # Run
        st.markdown("## Output")
        PrintWidget = st.empty()
        args = NLP_ARGS(dataset, out_folder, segmenter, tokenizer, custom, printObj=printObj)
        searchEngine = SearchEngine(args)
        # Either handle query from user or evaluate on the complete dataset
        if args.custom:
            searchEngine.handleCustomQuery(query)
        else:
            searchEngine.evaluateDataset()
            st.markdown("## Evaluation")
            st.image(os.path.join(out_folder, PATH_EVALPLOT), caption="Evaluation", use_column_width=True)
        


#############################################################################################################################
# Driver Code
if __name__ == "__main__":
    PRINT_OBJ = printObj
    main()