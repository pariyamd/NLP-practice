import re
import scipy.optimize as optimize

c = 500
unigram = {}
unigram_prob = {}
bigram = {}
bigram_prob = {}
trigram = {}
trigram_prob = {}
# global num_of_words
num_of_words = 0


def back_off(w2, w1, w0, l2, l1, l0):
    b = unigram_prob[w0] * l0
    if (w1, w0) in bigram.keys():
        b += bigram_prob[(w1, w0)] * l1
    if (w2, w1, w0) in trigram.keys():
        b += trigram_prob[(w2, w1, w0)] * l2
    return b


def guess(line, l2, l1, l0):
    w2 = '<s>'
    w1 = '<s>'
    m = 0
    word = None
    for w in line.split():
        if w == '$':
            for k in unigram.keys():
                a = back_off(w2, w1, k, l2, l1, l0)
                if m < a:
                    m = a
                    word = k
            break
        w2 = w1
        w1 = w
    return word


def unigram_cal(line):
    global num_of_words
    for word in line.split():
        num_of_words += 1
        if word in unigram.keys():
            unigram[word] += 1
        else:
            unigram[word] = 1
    return unigram_prob


def bigram_cal(line):
    w1 = '<s>'
    for w in line.split():
        if (w1, w) in bigram.keys():
            bigram[(w1, w)] += 1
        else:
            bigram[(w1, w)] = 1
        w1 = w
    return bigram_prob


def trigram_cal(line):
    w1 = '<s>'
    w2 = '<s>'
    for w in line.split():
        if (w2, w1, w) in trigram.keys():
            trigram[(w2, w1, w)] += 1
        else:
            trigram[(w2, w1, w)] = 1
        w2 = w1
        w1 = w
    return trigram


with open('Train_data.rtf', 'r') as text:
    my_txt = []
    for line in text:
        c -= 1
        if c == 0:
            break
        l = re.compile("^[^\\\\].*\.\\\\$")
        for i in l.findall(line):
            n = i.rfind('.')
            compiled = "<s> " + i[:n] + " </s>"
            my_txt.append(compiled)
            unigram_prob = unigram_cal(compiled)

for k in unigram:
    unigram_prob[k] = unigram[k] / num_of_words
# for k, v in unigram.items():
#     print(k, ':', v)

for line in my_txt:
    bigram_prob = bigram_cal(line)
for k in bigram.keys():
    bigram_prob[k] = bigram[k] / unigram[k[0]]
# for k, v in bigram_prob.items():
#     print(k, ' : ', v)

for line in my_txt:
    trigram_prob = trigram_cal(line)
for k in trigram.keys():
    trigram_prob[k] = trigram[k] / bigram[(k[0], k[1])]
# for k, v in trigram_prob.items():
#     print(k, ' : ', v)
with open('Test_data.rtf', 'r') as text:
    my_tst = []
    for line in text:
        print(line)
        l = re.compile(r'[0-9]*,+[\\*\'*\"*[0-9]*]*(.*)')
        comp = l.findall(line)
        if comp:
            n = comp[0].rfind(".")
            compiled_line = comp[0][:n]
            my_tst.append(compiled_line)
my_tst.pop(0)
print(len(my_tst))
labs = []
i = 0
with open('labels.rtf', 'r') as text:
    for line in text:
        l = re.compile(r'[0-9]*,+(.*)\\')
        lab = l.findall(line)
        if lab:
            lab = lab[0].strip()
            labs.append(lab)
            # guessed = guess(my_tst[i])
            # print("m", lab, "g", guessed, "\t\t", guessed == lab)
            # i += 1


#
#
def f(landas):
    wrong = 0
    for i in range(len(labs)):
        guessed = guess(my_tst[i], landas[0], landas[1], landas[2])
        print(my_tst[i])
        print(guessed, "->", labs[i])
        if guessed != labs[i]:
            wrong += 1
    # print(landas, wrong)
    return wrong


f([1, 1, 1])

#
# initial_guess = [0, 0, 0]
# result = optimize.minimize(f, initial_guess)
#
# if result.success:
#     fitted_params = result.x
#     print(fitted_params)
# else:
#     raise ValueError(result.message)
