checkpoint: checkpoints\tiny_gpt.pt  (809,984 params)
full validation set: 111,540 chars, loss 1.6348 nats/char, perplexity 5.128, 2.359 bits/char

### Prompt 1: 'ROMEO:'  (temperature 0.8, seed 0)
ROMEO:
Why make the prevence of your any, in I to thee,
For to givent be receity to be the senself.

RICHARD:
I pray thee, and this did trunk; my father well.

RIVERS:
He me to the hear and much aut no.

SICINIUS:
Thanks is thine not say that of me;
Or my 

### Prompt 2: 'JULIET:\nO '  (temperature 0.8, seed 1)
JULIET:
O my word here to unage, that me, by my shall with a virtue,
And he mother mine, and long her, we shall with the
Too was hither in a sishouts there: place your head
many my bark with thou the fair three with my pass,
And I do territor it whipted desire

### Prompt 3: 'KING HENRY:\nWhat news'  (temperature 0.8, seed 2)
KING HENRY:
What news do my lords, her you to speak of the kissning.

DUKE VINCENTIO:
How so the batts with what he with the fault,
By the may set that dank the sigh fast,
And say my chargelign like made make with me.

WARWICK:
How no thou drawnesss in commondion of asta

### Prompt 4: 'First Citizen:\n'  (temperature 0.8, seed 3)
First Citizen:
Then is but say my liege, that I'll myself,
When think the like alther after the worth,
Thou art the Lord Vantage,
For small'd prots of the to penduation:
And we may hate lips to me justices in the break,
And, very hanged blings the doth word men ang

### Prompt 5: 'To be, or not to be'  (temperature 0.8, seed 4)
To be, or not to be it, with was
Yet do whose racks fair in the him life well
Have her hate speak be in the scorn a wrong
Till death all made,
And that I know, when that would hope have
Till cannot like the ead to orde look putcher;
The comes honour, that is the voice 

### Greedy vs temperature sampling (prompt 'ROMEO:', 400 new chars, mean over 3 seeds where stochastic)
strategy                distinct 4-grams  repeated-text frac
greedy                             0.249               0.600
temperature 0.5                    0.780               0.008
temperature 0.8                    0.871               0.004
temperature 1.0                    0.920               0.002
temperature 1.5                    0.973               0.000

--- greedy ---

What shall be the death the shall be the see
That the shall be the seem to the world the see
That the words the shall be the see to the words.

CORIOLANUS:
I will not the words the shall be the see
That world the shall be the shall be the see
That world the shall be the see to the world,
That will 

--- temperature 0.5 ---

Why the disconsel and of the hand, in the strength
As with the been again of our hearts.

Second Murderer:
Which we all the friends not more the ears,
That dare this hath a cheek and the death,
And the love prove the should be say the could
The of wing the mean of her pointed the hath country
With 

--- temperature 0.8 ---

Why make the prevence of your any, in I to thee,
For to givent be receity to be the senself.

RICHARD:
I pray thee, and this did trunk; my father well.

RIVERS:
He me to the hear and much aut no.

SICINIUS:
Thanks is thine not say that of me;
Or my inferity to the neighting of the powers?

JULIET:


--- temperature 1.0 ---

Why mhousands! Therefore ineach'd, in I tleep to come.

POLIXENES:
My father.

GREMIO:
These is stants, your wake a man: my let,
Rather my father well you for mine of the wintes.

MENENIUS:
Rich. So I fearful him fill my honours,
Tell come me being inle in me stouth,
Thou should one that frantly ha

--- temperature 1.5 ---

Porform'd her! Tlan, o'ca means! Paid Ighleve greech,
By his twaker? hath Doth O, even make him
stants, your wake exain: my art, trunk-days thy deven's
to firief Iimestod you! who nose wretul ne. So I
fearful him-milkem,--In take feign of lighbain.
And bird Romeo!

VOLUMNIA:
Fater! the stort! my ha

