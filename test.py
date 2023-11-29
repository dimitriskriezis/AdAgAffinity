# import torch
# import torch.nn as nn

# input1 = torch.randn(100, 128)
# input2 = torch.randn(100, 128)
# cos = nn.CosineSimilarity(dim=1, eps=1e-6)
# output = cos(input1, input2)
# print(output)
# print(len(output))

def getMaxAdditionalDinersCount(N: int, K: int, M: int, S) -> int:
  S.append(0)
  sortedS = sorted(S)
  sortedS.append(N+1)
  diners = 0
  for i in range(0, M+1):
    if i == 0 or i == M:
      diff = sortedS[i+1] - sortedS[i] - K -1
      
    else:
      diff = sortedS[i+1] - sortedS[i] - 2*K-1
    
    if diff > 0:
      print(diff, sortedS[i], sortedS[i+1])
      diners += (int((diff - 1)/(K+1)))+1

  return diners

a = getMaxAdditionalDinersCount(15, 2, 3, [11, 14, 6])
print(a)
