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
                    help="English output from a standard Visual Encoder-\
                    Decoder model, one sentence per line. Required.")
parser.add_argument("--ger_mono", required=True,
                    help="German output from a standard Visual Encoder-\
                    Decoder model, one sentence per line. Required.")

parser.add_argument("--eng_multi",
                    help="English output from a German Visual Encoder\
                    - English Visual Decoder model. One sentence per line.\
                    Optional.")
parser.add_argument("--ger_multi",
                    help="German output from an English Visual Encoder\
                    - German Visual Decoder model. One sentence per line.\
                    Optional.")

parser.add_argument("--eng_multi_noimage",
                    help="English output from a German Visual Encoder\
                    - English MT + Source Decoder model. One sentence per\
                    line. Optional.")
parser.add_argument("--ger_multi_noimage",
                    help="German output from an English Visual Encoder\
                    - German MT + Source Decoder model. One sentence per\
                    line. Optional.")

parser.add_argument("--eng_mtenc_mtdec",
                    help="English output from a MT Seq-Seq model,\
                    one per line. Optional.")
parser.add_argument("--ger_mtenc_mtdec",
                    help="German output from a MT Seq-Seq model,\
                    one per line. Optional.")

parser.add_argument("--eng_mtenc_visdec",
                    help="English output from a German MT Encoder and an\
                    English Visual Decoder, one per line. Optional.")
parser.add_argument("--ger_mtenc_visdec",
                    help="German output from an English MT Encoder and a\
                    German Visual Decoder, one per line. Optional.")

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

if args.eng_multi_noimage is not None:
    eng_multi_noimage = open(args.eng_multi_noimage).read().split("\n")
else:
    eng_multi_noimage = None

if args.eng_mtenc_mtdec is not None:
    eng_mtenc_mtdec = open(args.eng_mtenc_mtdec).read().split("\n")
else:
    eng_mtenc_mtdec = None

if args.eng_mtenc_visdec is not None:
    eng_mtenc_visdec = open(args.eng_mtenc_visdec).read().split("\n")
else:
    eng_mtenc_visdec = None

ger_ref = codecs.open(args.ger_ref, 'r', 'utf-8').read().split("\n")
ger_mono = codecs.open(args.ger_mono, 'r', 'utf-8').read().split("\n")

if args.ger_multi:
    ger_multi = codecs.open(args.ger_multi, 'r', 'utf-8').read().split("\n")
else:
    ger_multi = None

if args.ger_multi_noimage is not None:
    ger_multi_noimage = open(args.ger_multi_noimage).read().split("\n")
else:
    ger_multi_noimage = None

if args.ger_mtenc_mtdec is not None:
    ger_mtenc_mtdec = open(args.ger_mtenc_mtdec).read().split("\n")
else:
    ger_mtenc_mtdec = None

if args.ger_mtenc_visdec is not None:
    ger_mtenc_visdec = open(args.ger_mtenc_visdec).read().split("\n")
else:
    ger_mtenc_visdec = None


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

doc = dominate.document(title='Validation dataset visualisation')
with doc.head:
    link(rel='stylesheet', href='style.css')
    meta(http_equiv="Content-Type", content="text/html; charset=utf-8")

handle = codecs.open("val.html", "w", "utf-8")
with doc:
    key = div(cls="text")
    key.add(p("<b>MLM</b>: Multimodal Language Model"))
    key.add(p("<b>LM</b>: Standard Language Model"))
    for idx, img_name in enumerate(image_names):
        with div(cls='instance'):
            h3(img_name)
            # Render the English content
            left = div(cls="text")
            unkified = replace_unk(eng_ref[idx], eng_vocab)
            left.add(p("<b>Reference:</b> %s" % unkified))
            left.add(p("<b>MLM:</b> %s" % eng_mono[idx]))
            if eng_multi:
                left.add(p("<b>MLM -> MLM:</b> %s" % eng_multi[idx]))
            if eng_multi_noimage:
                left.add(p("<b>MLM -> LM:</b> %s" % eng_multi_noimage[idx]))
            if eng_mtenc_mtdec:
                left.add(p("<b>LM -> LM:</b> %s" % eng_mtenc_mtdec[idx]))
            if eng_mtenc_visdec:
                left.add(p("<b>LM -> MLM:</b> %s" % eng_mtenc_visdec[idx]))

            # Render the image
            # BUG: this URL will most likely be wrong because it takes the
            #      absolute path to the image, as defined by the content in
            #      --image_names.
            img(src=img_name)

            # Render the German content
            right = div(cls="text")
            unkified = replace_unk(ger_ref[idx], ger_vocab)
            right.add(p("<b>Reference:</b> %s" % unkified))
            right.add(p("<b>MLM:</b> %s" % ger_mono[idx]))
            if ger_multi:
                right.add(p("<b>MLM -> MLM:</b> %s" % ger_multi[idx]))
            if ger_multi_noimage:
                right.add(p("<b>MLM -> LM:</b> %s" % ger_multi_noimage[idx]))
            if ger_mtenc_mtdec:
                right.add(p("<b>LM -> LM:</b> %s" % ger_mtenc_mtdec[idx]))
            if ger_mtenc_visdec:
                right.add(p("<b>LM -> MLM:</b> %s" % ger_mtenc_visdec[idx]))

content = doc.render()
content = content.replace("&lt;", "<")
content = content.replace("&gt;", ">")
handle.write(content)

handle.close()
