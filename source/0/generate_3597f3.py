# https://github.com/davidwrenner/Qiskit-Gate-Art/blob/b210f9de6108fa98a454b1864e4f93ff0f71305e/generate.py
#!/usr/bin/env python3

import random
import sys
from PIL import Image, ImageFont, ImageDraw
from qiskit import QuantumCircuit

DPI = 250
FONT_SIZE = 19
FONT_FILE = "SourceSansPro-Bold.otf"
OUTPUT_MODE = "mpl"
OUTPUT_FILE = "output.png"
MAX_TEXT_LENGTH = 200
H_COLOR = "#A51417"
I_COLOR = "#F0F0F0"
X_COLOR = "#F0F0F0"
Y_COLOR = "#F0F0F0"
Z_COLOR = "#F0F0F0"


class TextCircuit:
    def __init__(self, text):
        self.text = text
        self.font = ImageFont.truetype(FONT_FILE, FONT_SIZE)
        _, _, self.w, self.h = self.font.getbbox(self.text)
        self.image = self.init_image()
        self.offset = self.compute_offset()
        self.qc = QuantumCircuit(self.h - self.offset)
        self.image_to_qc()

    def image_to_qc(self):
        for r in range(self.offset, self.h):
            for c in range(self.w):
                self.bit_to_gate(r, c)

    def compute_offset(self):
        r = 0
        while all([self.image.getpixel((c, r)) for c in range(self.w)]):
            r += 1
        return r

    def bit_to_gate(self, r, c):
        if self.image.getpixel((c, r)):
            self.append_random(r - self.offset)
        else:
            self.qc.h(r - self.offset)

    def append_random(self, qubit):
        n = random.uniform(0, 1)
        if n < 0.25:
            self.qc.i(qubit)
        elif n < 0.5:
            self.qc.x(qubit)
        elif n < 0.75:
            self.qc.y(qubit)
        else:
            self.qc.z(qubit)

    def init_image(self):
        image = Image.new("1", (self.w, self.h), 1)
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), self.text, font=self.font)
        return image

    def save(self):
        self.qc.draw(output=OUTPUT_MODE,
                     filename=OUTPUT_FILE,
                     fold=MAX_TEXT_LENGTH,
                     style={
                         "displaycolor": {
                             "h": H_COLOR,
                             "id": I_COLOR,
                             "x": X_COLOR,
                             "y": Y_COLOR,
                             "z": Z_COLOR
                         },
                         "dpi": DPI
                     })


def usage():
    print(f"usage: {sys.argv[0]} text")
    print(f"   Maximum text length: {MAX_TEXT_LENGTH}\n")


def get_input_text():
    input_text = ""
    for i in range(1, len(sys.argv)):
        input_text += sys.argv[i] + " "
    return input_text.rstrip()


def main():
    input_text = get_input_text()
    if not input_text:
        usage()
        return
    if len(input_text) > MAX_TEXT_LENGTH:
        raise IOError(f"Length of input ({len(input_text)}) exceeded ({MAX_TEXT_LENGTH})")
    tc = TextCircuit(input_text)
    tc.save()


if __name__ == "__main__":
    main()
