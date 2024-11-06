class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 # input_embeddings,
                 label,

                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        # self.input_embeddings = input_embeddings
        self.label=label