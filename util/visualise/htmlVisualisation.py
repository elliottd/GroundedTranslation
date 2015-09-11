import dominate
from dominate.tags import *
import argparse
import codecs
import pickle

parser = argparse.ArgumentParser("Generate an HTML file visualising the\
                                  output of different models")
parser.add_argument("--images", required=True,
                    help="Image file names, one per line. Required.")
parser.add_argument("--eng_vocab", required=True,
                    help="English vocabulary pickle. Required.")
parser.add_argument("--ger_vocab", required=True,
                    help="German vocabulary pickle. Required.")
parser.add_argument("--eng_ref", required=True,
                    help="English references, one per line. Required.")
parser.add_argument("--ger_ref", required=True,
                    help="German references, one per line. Required.")
parser.add_argument("--eng_mono", required=True,
                    help="Monolingual English output, one per line.\
                          Required.")
parser.add_argument("--ger_mono", required=True,
                    help="Monolingual German ouput, one per line.\
                    Required.")
parser.add_argument("--eng_multi", help="Multilingual English output, \
                                         one per line. Optional.")
parser.add_argument("--ger_multi", help="Multilingual German ouput, \
                                         one per line. Optional.")

args = parser.parse_args()

image_names = open(args.images).read().split()
eng_vocab = pickle.load(open(args.eng_vocab, "rb")).keys()
ger_vocab = pickle.load(open(args.ger_vocab, "rb")).keys()

eng_ref = open(args.eng_ref).read().split("\n")
eng_mono = open(args.eng_mono).read().split("\n")

if args.eng_multi is not None:
    eng_multi = open(args.eng_multi).read().split("\n")
else:
    eng_multi = None

ger_ref = codecs.open(args.ger_ref, 'r', 'utf-8').read().split("\n")
ger_mono = codecs.open(args.ger_mono, 'r', 'utf-8').read().split("\n")

if args.ger_multi:
    ger_multi = codecs.open(args.ger_multi, 'r', 'utf-8').read().split("\n")
else:
    ger_multi = None


def replace_unk(words, vocab):
    '''
    Add UNK CSS markings into a string for visualisation.
    '''
    unkified = ""
    for x in words.split():
        if x not in vocab:
            unkified += "<span class='unk'>%s</span> " % x
        else:
            unkified += "%s " % x
    return unkified[:-1]

doc = dominate.document(title='Validation data set visualisation')
with doc.head:
    link(rel='stylesheet', href='style.css')
    meta(http_equiv="Content-Type", content="text/html; charset=utf-8")

handle = codecs.open("val.html", "w", "utf-8")
with doc:
    for idx, img_name in enumerate(image_names[0:100]):
        with div(cls='instance'):
            h3(img_name)
            # Render the English content
            left = div(cls="text")
            unkified = replace_unk(eng_ref[idx], eng_vocab)
            left.add(p("<b>Ref:</b> %s" % unkified))
            left.add(p("<b>Mono:</b> %s" % eng_mono[idx]))
            if eng_multi:
                left.add(p("<b>Multi:</b> %s" % eng_multi[idx]))

            # Render the image
            # BUG: this URL will most likely be wrong because it takes the
            #      absolute path to the image, as defined by the content in
            #      --image_names.
            img(src=img_name)

            # Render the German content
            right = div(cls="text")
            unkified = replace_unk(ger_ref[idx], ger_vocab)
            right.add(p("<b>Ref:</b> %s" % unkified))
            right.add(p("<b>Mono:</b> %s" % ger_mono[idx]))
            if ger_multi:
                right.add(p("<b>Multi:</b> %s" % ger_multi[idx]))

content = doc.render()
content = content.replace("&lt;", "<")
content = content.replace("&gt;", ">")
handle.write(content)

handle.close()
