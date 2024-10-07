# ML_midterm_assignment

## Task
Project description: Given a set of truncated texts, for each piece of text xi, such as “Yesterday I went”, ask different Large Language Models (LLMs) to complete it by appending xj =”to Costco and purchased a floor cleaner.” so you get a complete text like “Yesterday I went to Costco and purchased a floor cleaner.” from each LLM. The same xi leads to different xj. Now please build a deep learning classifier to figure out, for each input (xi, xj), which LLM was used for this pair.

## Code Execution

To run the files and replicate the results, run BLOOM.py, CTRL.py, GPT2.py, OPT.py and T5.py. 5 datasets will be generated. This will contain the respective xj for the xi and the Combined sentance(xi+xj). To train the classifier only the Combined sentance is required. So all the 5 are combined and stored in Filtered.xlsx. The classifier code is present in the Training.ipynb file. Running this will give the accuracy of the classifier when you consider all the 5 LLMs as well as a combination of 4 LLMs.
