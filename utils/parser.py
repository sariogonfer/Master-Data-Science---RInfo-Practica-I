import os

from bs4 import BeautifulSoup as bs4


def html2txt_parser(html_path, txt_path):
    with open(html_path, "rb") as in_, open(txt_path, 'a') as out:
        bs = bs4(in_.read())
        [out.write(p.get_text()) for p in bs.find_all('p')]

def html2txt_parser_dir(in_path, out_path=None):
    out_path = out_path if out_path else in_path
    for html in [h for h in os.listdir(in_path) if h.endswith('.html')]:
        html2txt_parser(os.path.join(in_path, html),
                        os.path.join(out_path, html.replace('html', 'txt')))
