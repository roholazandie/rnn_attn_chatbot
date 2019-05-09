## Chatbot using RNN and Luong Attention in pytorch
This is a repository based on original pytorch tutorial on creating a chatbot using pytorch


## Dataset
Dataset is availabe [here](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
Extract and put it in a directory named "cornell_movie_dialogs_corpus" besides the code.


### prepare data
```
python prepare_data --corpus-dir cornell_movie_dialogs_corpus --corpus-file movie_lines.txt --conversations-corpus-file movie_conversations.txt
```

## How to run?
For training:
```
python run_model.py run_training --corpus-dir cornell_movie_dialogs_corpus --save-dir cornell_movie_dialogs_corpus/save --datafile cornell_movie_dialogs_corpus/formatted_movie_lines.txt --config-file config.json
```

For evaluation(interactive chatbot):
```
python run_evaluation --corpus-dir cornell_movie_dialogs_corpus --save-dir cornell_movie_dialogs_corpus/save --datafile cornell_movie_dialogs_corpus/formatted_movie_lines.txt --config-file config.json
```

