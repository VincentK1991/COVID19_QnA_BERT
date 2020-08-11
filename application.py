from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_table
from textwrap import dedent

import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
import numpy as np
import pandas as pd

import nltk
nltk.download('punkt')

nltk.download('stopwords')
regex_tokenizer = nltk.RegexpTokenizer(r"[a-zA-Z]+")
stemmer = SnowballStemmer('english')
stopwords_set = set(stopwords.words('english'))


""" 
=============================================
INITIALIZATION
DO IN ONCE AND KEEP IT IN MEMOERY
=============================================
"""
# initialize and run stuff here before going to the app
print('initialize models')
# reload the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(
    'distilbert-base-uncased-distilled-squad')
bert_model = DistilBertForQuestionAnswering.from_pretrained(
    'distilbert-base-uncased-distilled-squad')
print('finished initializing BERT')

# reload the TFIDF
database = pd.read_csv('dataset/processed_metadata_Aug11_2020.csv')
database = database.fillna('')
tfidf_obj = TfidfVectorizer(max_features=5000)
data_TFIDF = tfidf_obj.fit_transform(database['new_abstract'])
print('finished setting up database')

# fit the nearest neighbors
nbrs = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='cosine')
nbrs.fit(data_TFIDF)
print('finished setting up retrieval mechanism')

with open('dataset/wiki.txt', 'r') as f:
    WIKI = f.read()
WIKI = WIKI.replace('\n', '')
print('finished loading WIKIPEDIA articles')
print('=======================================')
print(' ')

""" 
=============================================
HELPER FUNCTION SECTION
=============================================
"""


def text_to_query(text, tfidf_obj):
    """ 
    turn the text to TFIDF query
    """
    list_token = regex_tokenizer.tokenize(text)
    list_token = list(map(lambda x: x.lower(), list_token))
    list_token = list(filter(lambda x: x not in stopwords_set, list_token))
    list_token = list(map(lambda x: stemmer.stem(x), list_token))
    list_text = [' '.join(list_token)]
    return tfidf_obj.transform(list_text)


def query_index(neighbor_set, query, num_neighbors, threshold=0.4):
    """ use k-nearest neighbors with cosine similarity to find TFIDF documents that are closed 
    to the search query
    input: neighbor database (TFIDF format)
            TFIDF search query (TFIDF format as well)
            number of neighbors to be queried
            cosine similarity threshold applied after nearest neighbors
    output: array index pointing to the TFIDF database 
    """
    arr_dist, arr_idx = neighbor_set.kneighbors(
        query, num_neighbors, return_distance=True)
    return arr_idx[np.where(arr_dist < threshold)].tolist()


def bert_tokenize(query, frame, list_key):
    """
    tokenize the pair of question and context
    return input tokens
            attention_mask
    """
    list_text = frame.iloc[list_key]
    sample_size = len(list_text)
    list_query = [query]*sample_size
    list_token = []
    for item1, item2 in zip(list_query, list_text):
        list_token.append(tokenizer.encode(item1, item2))
    list_padded_token = []
    for item in list_token:
        item = item[:512] + [0]*(512-len(item))
        list_padded_token.append(item)

    mask_array = np.zeros((sample_size, 512))

    for count, item in enumerate(list_padded_token):
        # print(item)
        mask_array[count] = list(map(lambda x: 1 if x != 0 else 0, item))

    torch_token = torch.tensor(list_padded_token, dtype=torch.long)
    torch_mask = torch.tensor(mask_array)
    return torch_token, torch_mask


def bert_answering(input_token, input_mask):
    """ 
    takes the tokenized input and attention mask
    return the tuple of start position and end position of answers
    """
    bert_model.eval()
    with torch.no_grad():
        output = bert_model(input_ids=input_token,
                            start_positions=input_start)
        # return output
    answer_start = torch.argmax(output[0], axis=1).tolist()
    answer_end = torch.argmax(output[1], axis=1).tolist()
    return tuple(zip(answer_start, answer_end))


def decode_answer(input_token, tuple_span):
    """
    select the answer (if where the start position < end position)
    and start position != 0 (this is the default when answer is not found)

    return the list answer 
    """
    list_answer = []
    list_count = []
    for count, span in enumerate(tuple_span):
        if span[0] < span[1] and span[0] != 0:
            list_answer.append(tokenizer.decode(
                input_token[count, span[0]:span[1]+1]))
            list_count.append(count)
    return list_answer, list_count


def create_wiki_frame(list_answer):
    """
    create dataframe from the answer from wikipedia
    """
    list_url = [
        'https://en.wikipedia.org/wiki/Coronavirus_disease_2019']*len(list_answer)
    answer_dict = {'answer': list_answer, 'url': list_url}
    answer_frame = pd.DataFrame(answer_dict)
    return answer_frame


def create_answer_frame(list_answer, list_index, list_key, frame):
    """
    create dataframe from the list of answer for the COVID19 open research
    """
    absolute_index = [list_key[i] for i in list_index]
    list_url = frame.iloc[absolute_index].tolist()
    answer_dict = {'answer': list_answer, 'url': list_url}
    answer_frame = pd.DataFrame(answer_dict)
    return answer_frame


def sliding_windows(list_tokens, overlap=50, max_len=400):
    """ generate an overlap sliding partition of large text file
    to smaller text size (as input for BERT is limited to 512 tokens) """
    list_of_list = []
    list_of_position = []
    piece = 0

    list_of_list.append(list_tokens[0: (piece + max_len)])
    list_of_position.append((0, (piece + max_len - 1)))
    piece += max_len

    while piece < len(list_tokens):
        list_of_list.append(
            list_tokens[(piece - overlap): (piece + max_len - overlap)])
        list_of_position.append(
            (piece-overlap, min((piece + max_len - overlap - 1), len(list_tokens) - 1)))
        piece += max_len - overlap
    return list_of_list, list_of_position


def padding(item):
    """ create a padding for batch input, the pad token is [0] """
    return item + [0]*(512 - len(item))


def get_attention_masks(item):
    """create an attention masking where the non-word padding get masked by 0.0"""
    return list(map(lambda x: 1 if x != 0 else 0, item))


def tokens_from_quesion(question, context):
    """create a pair of question & context for wikipedia articles"""
    token_context = tokenizer.encode(context, add_special_tokens=False)
    token_question = tokenizer.encode(question, add_special_tokens=False)
    list_context, list_pos = sliding_windows(
        token_context, overlap=50, max_len=400)

    list_of_pairs = []
    for item_context in list_context:
        tokens = [tokenizer.cls_token_id] + token_question + \
            [tokenizer.sep_token_id] + item_context + [tokenizer.sep_token_id]
        list_of_pairs.append(tokens)

    list_padded_pairs = []
    for item in list_of_pairs:
        padded_item = padding(item)
        list_padded_pairs.append(padded_item)

    list_attention_masks = []
    for item in list_padded_pairs:
        mask_item = get_attention_masks(item)
        list_attention_masks.append(mask_item)
    return torch.tensor(list_padded_pairs, dtype=torch.long), torch.tensor(list_attention_masks, dtype=torch.float)


''' 
===================================
END OF THE HELPER FUNCTION SECTION
===================================
'''


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
application = app.server


PAGE_SIZE = 15
app.layout = html.Div([
    dcc.Textarea(
        id='textarea-state-example',
        value='replace this text with your question about COVID-19',
        style={'width': '100%', 'height': 200},
    ),
    html.Div([
        html.Div('''adjust threshold of retrieval''', style={
                 'paddingTop': 2.5, 'paddingBottom': 5}),
        dcc.Slider(id='my-slider', min=0, max=1, step=0.05, value=0.5,
                   marks={
                       0: {'label': '0', 'style': {'color': '#77b0b1'}},
                       0.2: {'label': '0.2'},
                       0.4: {'label': '0.4'},
                       0.6: {'label': '0.6'},
                       0.8: {'label': '0.8'},
                       1: {'label': '1.0', 'style': {'color': '#f50'}}
                   })
    ]),
    html.Button('Submit', id='textarea-state-example-button', n_clicks=0),
    dcc.RadioItems(id='my-toggle-switch',
                   options=[
                       {'label': 'Search WIKI', 'value': True},
                       {'label': 'Skip WIKI', 'value': False},
                   ],
                   value=False
                   ),
    html.Div([
        dcc.Markdown(dedent('''
            #
            #
            #### COVID19 Open Research Datset'''))]),
    html.Div([dash_table.DataTable(id='datatable-paging-page-count',
                                   columns=[{"name": i, "id": i}
                                            for i in ['answer', 'url']],
                                   page_current=0,
                                   page_size=PAGE_SIZE,
                                   page_action='custom',
                                   style_cell={
                                       'whiteSpace': 'normal',
                                       'height': 'auto'
                                   }
                                   )]),

    html.Div([
        dcc.Markdown(dedent('''
            #
            #
            #### WIKIPEDIA'''))]),

    html.Div([dash_table.DataTable(id='datatable-wiki',
                                   columns=[{"name": i, "id": i}
                                            for i in ['answer', 'url']],
                                   page_current=0,
                                   page_size=PAGE_SIZE,
                                   page_action='custom',
                                   style_cell={
                                       'whiteSpace': 'normal',
                                       'height': 'auto'
                                   }
                                   )]),
    html.Div(id='my-output-print'),

    html.Div([
        dcc.Markdown(dedent('''
            #
            #
            =====================================
            ## COVID19 question answering Web app
            ##### the data is collected from [COVID-19 Open Research Dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) and from [Wikipedia](https://en.wikipedia.org/wiki/Coronavirus_disease_2019)
            ##### Presented by Vincent Kieuvongngam [Vincent Github here](https://github.com/VincentK1991)


            Note that this web app aims at developing machine learning tools for interactive question answering of COVID19.
            
            If you find information useful please verify the formation by copy and paste the URL to your web browser.


            ###### This web app cannot provide medical advice, diadnosis or treatment of COVID19. 
            '''))
    ],
        style={"width": '60%', 'float': 'left', 'display': 'inline-block', 'padding': '0px 10px 150px 0px', 'boxSizing': 'border-box'})
])


@app.callback(
    [Output('datatable-paging-page-count', 'data'),
     Output('datatable-wiki', 'data'),
     Output('my-output-print', 'children')],
    [Input('textarea-state-example-button', 'n_clicks'),
     Input('my-slider', 'value'),
     Input('my-toggle-switch', 'value')],
    [State('textarea-state-example', 'value')]
)
def update_output(n_clicks, threshold, toggle, value):
    data = {'answer': ['...'], 'url': ['...']}
    df = pd.DataFrame.from_dict(data)
    if n_clicks > 0:
        query_text = text_to_query(value, tfidf_obj)
        if toggle:
            batch1 = tokens_from_quesion(value, WIKI)
            output = bert_model(input_ids=batch1[0], attention_mask=batch1[1])
            answer_start = torch.argmax(output[0], axis=1).tolist()
            answer_stop = torch.argmax(output[1], axis=1).tolist()
            tuple_span = tuple(zip(answer_start, answer_stop))
            list_answer, list_index = decode_answer(batch1[0], tuple_span)

            WIKI_frame = create_wiki_frame(list_answer)
            return df.to_dict('records'), WIKI_frame.to_dict('records'), 'we found {} answer'.format(len(WIKI_frame))
        else:
            list_keys = query_index(nbrs, query_text, 20, threshold=threshold)
            print(list_keys)

            if len(list_keys) == 0:
                data = {'answer': ['...'], 'url': ['placeholder']}
                df = pd.DataFrame.from_dict(data)
                return df.to_dict('records'), df.to_dict('records'), 'we cannot find an answer. Maybe try rewording or increase the threshold.'

            else:
                bert_input = bert_tokenize(
                    value, database['abstract'], list_keys)
                bert_output = bert_answering(
                    bert_input[0], bert_input[1])
                answer, list_index = decode_answer(bert_input[0], bert_output)
                print(list_index)
                if len(list_index) == 0:
                    return df.to_dict('records'), df.to_dict('records'), 'we cannot find an answer. Maybe try rewording or increase the threshold.'
                else:
                    result = create_answer_frame(
                        answer, list_index, list_keys, database['url'])
                    return result.to_dict('records'), df.to_dict('records'), 'we found {} answer'.format(len(list_index))

    if n_clicks == 0:
        return df.to_dict('records'), df.to_dict('records'), 'waiting fot the input submission.'


if __name__ == '__main__':

    application.run(debug=True)
