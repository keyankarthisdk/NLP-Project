"""
Stream lit GUI for hosting PyGames
"""

# Imports
import os
import streamlit as st
import json

import main

# Main Vars
PATH_EVALPLOT = "eval_plot.png"
LOGS = []
PrintWidget = None
ProgressWidget = {
    "title": None,
    "progress": None
}

# Main Classes
class NLP_ARGS:
    def __init__(self, dataset, out_folder, segmenter, tokenizer, custom, params={}, printObj=print):
        self.dataset = dataset
        self.out_folder = out_folder
        self.segmenter = segmenter
        self.tokenizer = tokenizer
        self.custom = custom
        self.params = params
        self.printObj = printObj

# Main Functions
# Object Functions
def printObj(text):
        global LOGS
        global PrintWidget
        LOGS.append(str(text))
        PrintWidget.text_area("Logs", "\n".join(LOGS), height=500)

def progressObj(key, value):
    global ProgressWidget
    ProgressWidget["title"].markdown(key)
    ProgressWidget["progress"].progress(value)

# Params Functions
def UI_Paths():
    dataset = st.text_input("Dataset Path", "cranfield/")
    out_folder = st.text_input("Output Folder Path", "output/")
    return dataset, out_folder

def UI_Segmenter():
    segmenter = st.selectbox("Choose Segmenter", ["naive", "punkt"])
    return segmenter

def UI_Tokenizer():
    tokenizer = st.selectbox("Choose Tokenizer", ["naive", "ptb"])
    params = {
        "ngram_n": st.number_input("Ngram N (1 for default unigram)", min_value=1, max_value=3, value=2, step=1)
    }
    return tokenizer, params

def UI_CustomQuery():
    custom = st.checkbox("Custom Query?")
    query = ""
    if custom:
        query = st.text_input("")
    return custom, query

# Runner Functions
def app_main():
    global LOGS
    global PrintWidget
    global ProgressWidget
    # Titles
    st.title("NLP Project")
    st.markdown("## Group 10")
    
    # Inputs
    dataset, out_folder = UI_Paths()
    segmenter = UI_Segmenter()
    tokenizer, tokenizer_params = UI_Tokenizer()
    custom, query = UI_CustomQuery()

    # Print Obj
    if st.button("Run"):
        LOGS = []
        
        # Run
        st.markdown("## Output")
        # Init Objects
        PrintWidget = st.empty()
        ProgressWidget = {
            "title": st.sidebar.empty(),
            "progress": st.sidebar.progress(0)
        }
        # Form args
        params = {}
        for p in [tokenizer_params]: params.update(p)
        args = NLP_ARGS(dataset, out_folder, segmenter, tokenizer, custom, params, printObj=printObj)
        searchEngine = main.SearchEngine(args)
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
    # Assign Objects
    main.PRINT_OBJ = printObj
    main.PROGRESS_OBJ = progressObj
    # Run Main
    app_main()