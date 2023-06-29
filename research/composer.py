h_first = 0xAC00
h_last = 0xD7A3

chosung = [chr(i) for i in range(0x1100, 0x1100 + 19)]
jungsung = [chr(i) for i in range(0x1161, 0x1161 + 21)]
jongsung = [''] + [chr(i) for i in range(0x11A8, 0x11A8 + 27)]

jongsung_cnv = [
	0x0000, 0x3131, 0x3132, 0x3133, 0x3134, 0x3135, 0x3136, 0x3137,
	0x3139, 0x313A, 0x313B, 0x313C, 0x313D, 0x313E, 0x313F, 0x3140,
	0x3141, 0x3142, 0x3144, 0x3145, 0x3146, 0x3147, 0x3148, 0x314A,
	0x314B, 0x314C, 0x314D, 0x314E,
]

def decompose(text, convert_to_jongsung=True):
	o = ''
	for c in text:
		a = ord(c)
		if a >= h_first and a <= h_last:
			i = a - h_first
			o += chosung[i // (len(jongsung) * len(jungsung))]
			o += jungsung[(i // len(jongsung)) % len(jungsung)]
			o += jongsung[i % len(jongsung)]
		elif convert_to_jongsung and a in jongsung_cnv:
			idx = jongsung_cnv.index(a)
			o += jongsung[idx]
		elif a > 0xFF:
			o += chr(0x2460 + a//256)
			o += chr(0x2460 + a%256)
		else:
			o += c
	return o
