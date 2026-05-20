#### LLAMA3 Inference
1. Reviewing the script I found that I was comparing apples to oranges. I was comparing llama2 instruct model with llama3 base model for inference and that is why llama3 inference results are horrific.
2. I am now going through the results with a fine-tooth comb. 
    * Rouge score will look at each word-word and often it does not match 
    e.g  GT: American Express's stock price has been on a consistent upward trend in the past weeks. This indicates strong investor belief in the company's future performance.                         
  Answer: AXP's stock price has been steadily increasing, reaching all-time highs, indicating investor confidence in the company's performance.         
    Wonder what is ther rouge score for this ?
    {'rouge1': Score(precision=0.6, recall=0.4444444444444444, fmeasure=0.5106382978723405), 'rouge2': Score(precision=0.3684210526315789, recall=0.2692307692307692, fmeasure=0.3111111111111111), 'rougeL': Score(precision=0.6, recall=0.4444444444444444, fmeasure=0.5106382978723405)}

    Rouge is worse on many level, word by word matching is an issue if model tries to speak in more than limited words about it, it can skip a line.

    

    GT: The acquisition deal between Discover Financial Services and Capital One can potentially
  lead
    to increased market competition, highlighting the advantage of American Express's stable      
  market
     position.                                                                                    
  Answer: AXP's operating margin and net margin have been stable, indicating its ability to
    maintain profitability.
  ────────────────────────────────────────
  #: 4                                        
  GT: —                                   
  Answer: The company has been included in the portfolios of prominent investors, such as Mario
    Gabelli, which suggests that it is a well-regarded investment opportunity.     

This answer is actually good by incorporating information other than the prompt and understanding what is happening but maybe missing from gt.

Wrong  Answer since cash ratio is less than 1.

The company's financials show a strong cash position, with a cash ratio of 0.3199 and a
    current ratio of 0.7395, indicating its ability to meet its short-term obligations. 


  GT: Up by 2-3%                                                                                  
  Answer: AXP's stock price is likely to continue its upward trend, reaching a high of 215.00 by
    the end of the next week (2024-03-03).

    Again wrong, base price was 213.9 so 2-3 percent would be close to 217-220 and 215 is 0.5 percent.

3. Write down the analysis here, qualitative.