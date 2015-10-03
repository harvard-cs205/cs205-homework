def Generate_New_Phrases(Word1_Word2, rdd):
  '''
  Generate New Random Phrases with 20 words given randomly-chosed Word1_Word2
  '''
  Phrases = list(Word1_Word2)
  for i in range(18):
    if i == 0:
      New_Word1_Word2 = Word1_Word2
    # Look up [(Word3, Count3)] for (Word1, Word2)
    Word3_Count3 = rdd.map(lambda x : x).lookup(New_Word1_Word2)[0]
    Word3_Count3.sort(key=lambda (word3, count3): count3)
    Chosen_Word3_Count3 = Word3_Count3[-1]
    Chosen_Word3 = Chosen_Word3_Count3[0]
    New_Word1_Word2 = (New_Word1_Word2[1], Chosen_Word3)
    Phrases.append(Chosen_Word3)

  return " ".join(Phrases)
