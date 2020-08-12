import re
from scipy import optimize

unigram = {}
unigram_prob = {}
bigram = {}
bigram_prob = {}
trigram = {}
trigram_prob = {}
num_of_words = 0


def back_off(w2, w1, w0, w3, w4, l2, l1, l0):
    b = 0
    if w0 in unigram_prob.keys():
        b = unigram_prob[w0] * l0

    if (w1, w0) in bigram.keys():
        b += bigram_prob[(w1, w0)] * l1

    if (w2, w1, w0) in trigram.keys():
        b += trigram_prob[(w2, w1, w0)] * l2

    return b


def guess(line, l2, l1, l0):
    w2 = '<s>'
    w1 = '<s>'
    w3 = '<s>'
    w4 = '<s>'
    m = 0
    words = line.split()
    word = None
    for i in range(len(words)):
        if words[i] == '$':
            if i+1 < len(words):
                w3 = words[i + 1]
            if i+2 < len(words):
                w4 = words[i + 2]
            for k in unigram.keys():

                a = back_off(w2, w1, k, w3,w4, l2, l1, l0)
                if m < a:
                    m = a
                    word = k
            break
        w2 = w1
        w1 = words[i]
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
        l = re.compile("^[^\\\\].*\.\\\\$")
        for i in l.findall(line.lower()):
            n = i.rfind('.')
            compiled = "<s> " + i[:n] + " </s>"
            my_txt.append(compiled)
            unigram_prob = unigram_cal(compiled)

for k in unigram:
    unigram_prob[k] = unigram[k] / num_of_words


for line in my_txt:
    bigram_prob = bigram_cal(line)
for k in bigram.keys():
    bigram_prob[k] = bigram[k] / unigram[k[0]]


for line in my_txt:
    trigram_prob = trigram_cal(line)
for k in trigram.keys():
    trigram_prob[k] = trigram[k] / bigram[(k[0], k[1])]


# read test data
with open('Test_data.rtf', 'r') as text:
    my_tst = []
    for line in text:
        l = re.compile(r'[0-9]*,+[\\*\'*\"*[0-9]*]*(.*)')
        comp = l.findall(line.lower())
        if comp:
            n = comp[0].rfind(".")
            compiled_line = comp[0][:n]
            my_tst.append(compiled_line)
my_tst.pop(0)
print(len(my_tst))

# read answers
labs = []
i = 0
with open('labels.rtf', 'r') as text:
    for line in text:
        l = re.compile(r'[0-9]*,+(.*)\\')
        lab = l.findall(line.lower())
        if lab:
            lab = lab[0].strip()
            labs.append(lab)


# cost function : number of wrong guesses
def f(landas):
    wrong = 0
    for i in range(len(labs)):
        guessed = guess(my_tst[i], landas[0], landas[1], landas[2])
        if guessed != labs[i]:
            wrong += 1
    print(landas, wrong)
    return wrong


initial_guess = [0.3,0.3,0.3]

# Says one minus the sum of all variables must be zero
cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
# Required to have non negative values
bnds = tuple((0, 1) for x in initial_guess)
result = optimize.minimize(f, initial_guess, method='SLSQP', bounds=bnds, constraints=cons)

if result.success:
    fitted_params = result.x
    print(fitted_params)
else:
    raise ValueError(result.message)
