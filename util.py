names = ["Helsinki-NLP/opus-mt-cs-en",
    "facebook/nllb-200-3.3B",
    "facebook/nllb-200-1.3B",
    "facebook/nllb-200-distilled-600M",
    "utter-project/EuroLLM-1.7B",
    "utter-project/EuroLLM-9B"]


# print intermidiate data of input, output and cross attention
def print_inp_out(inp_ids, dec_inp_ids, outputs, cross_att):
    print(inp_ids.shape, dec_inp_ids.shape)
    print(outputs.keys())
    print(type(cross_att), type(cross_att[0]), [[y.shape[2] for y in x] for x in cross_att])
    print(cross_att[-1][0].shape)