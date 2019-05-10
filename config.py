import json


class Config(object):

    def __init__(self,
                 hidden_size=500,
                 encoder_n_layers=2,
                 decoder_n_layers=2,
                 dropout=0.1,
                 batch_size=64,
                 checkpoint_iter= 4000,
                 attn_model= "dot",
                 model_name="cb_model",
                 load_filename=None,
                 clip=50.0,
                 teacher_forcing_ratio=1.0,
                 learning_rate=0.0001,
                 decoder_learning_ratio=5.0,
                 n_iterations=4000,
                 n_epochs=10,
                 print_every=1,
                 save_every=500,
                 corpus_name="",
                 bpe_codes_path="",
                 bpe_vocab_path="",
                 MAX_LENGTH=10,
                 MIN_COUNT=3,
                 device="cpu"
                 ):
        self.hidden_size = hidden_size
        self.encoder_n_layers = encoder_n_layers
        self.decoder_n_layers = decoder_n_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.checkpoint_iter = checkpoint_iter
        self.attn_model = attn_model
        self.model_name = model_name
        self.load_filename = load_filename
        self.clip = clip
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.learning_rate = learning_rate
        self.decoder_learning_ratio = decoder_learning_ratio
        self.n_iterations = n_iterations
        self.n_epochs = n_epochs
        self.print_every = print_every
        self.save_every = save_every
        self.corpus_name = corpus_name
        self.bpe_codes_path = bpe_codes_path
        self.bpe_vocab_path = bpe_vocab_path
        self.MAX_LENGTH = MAX_LENGTH
        self.MIN_COUNT = MIN_COUNT
        self.device = device

    @classmethod
    def from_dict(cls, json_object):
        config = Config(hidden_size=0)
        for key in json_object:
            config.__dict__[key] = json_object[key]
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file) as f:
            config_json = f.read()

        return cls.from_dict(json.loads(config_json))
