#include <stdio.h>
#include <math.h>

bool checkAccuracy(int your_answer, int ref_answer)
{
  int diff = your_answer - ref_answer;

  if (diff == 0) {
    // printf("your answer is correct\n");
    return true;
  } else {
    // printf("your answer is not correct: your answer is %d, right answer is %d\n", your_answer, ref_answer);
    return false;
  }
  
}