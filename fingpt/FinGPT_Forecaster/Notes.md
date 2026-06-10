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

#### llama3 instruct base model on masking numbers 

Around 50 percent of the output of the model contains the masked string for NUM in the generated text like this 

It is almost know in what pattern the analysis or findings should be presented, the real question is are there insights on how this number is generated. - can link COT here.

" prompt: For more expert insight and the latest market action, click here to watch this full episode of Yahoo Finance Live. Editor's note: This article was written by Nicholas Jacobino\n\nSome recent basic financials of AXP, reported at [NUM][NUM][NUM] are presented below:\n\n[Basic Financials]:\n\nassetTurnoverTTM: [NUM]\nbookValue: [NUM]\ncashRatio: [NUM]\ncurrentRatio: [NUM]\nebitPerShare: [NUM]\neps: [NUM]\nev: [NUM]\nfcfMargin: [NUM]\nfcfPerShareTTM: [NUM]\ngrossMargin: [NUM]\nlongtermDebtTotalAsset: [NUM]\nlongtermDebtTotalCapital: [NUM]\nlongtermDebtTotalEquity: [NUM]\nnetDebtToTotalCapital: [NUM]\nnetDebtToTotalEquity: [NUM]\nnetMargin: [NUM]\noperatingMargin: [NUM]\npayoutRatioTTM: [NUM]\npb: [NUM]\npeTTM: [NUM]\npfcfTTM: [NUM]\npretaxMargin: [NUM]\npsTTM: [NUM]\nquickRatio: [NUM]\nreceivablesTurnoverTTM: [NUM]\nroaTTM: [NUM]\nroeTTM: [NUM]\nroicTTM: [NUM]\nrotcTTM: [NUM]\nsalesPerShare: [NUM]\nsgaToSale: [NUM]\ntotalDebtToEquity: [NUM]\ntotalDebtToTotalAsset: [NUM]\ntotalDebtToTotalCapital: [NUM]\ntotalRatio: [NUM]\n\nBased on all the information before [NUM][NUM][NUM] let's first analyze the positive developments and potential concerns for AXP. Come up with [NUM][NUM] most important factors respectively and keep them concise. Most factors should be inferred from company related news. Then make your prediction of the AXP cryptocurrency price movement for next week ([NUM][NUM][NUM] to [NUM][NUM][NUM]). Provide a summary analysis to support your prediction.<|eot_id|><|start_head "

"generated": "[Positive Developments]:\n1. American Express and American Express Global Business Travel launched a new integration to streamline spend management for small businesses, enhancing their services and increasing revenue potential.\n2. The company declared a dividend, indicating a commitment to rewarding shareholders and providing a stable income stream.\n3. American Express emphasized the importance of generative AI in its annual letter to shareholders, signifying a focus on innovation and customer experience.\n4. The company's stock price has increased due to strong retail sales and credit card purchases, indicating a positive trend in the company's financial performance.\n5. American Express was featured as a top growth stock for the long-term, highlighting its potential for long-term growth and investment.\n\n[Potential Concerns]:\n1. The company's high debt levels, with a net debt-to-total equity ratio of [NUM], may pose a risk to its financial stability.\n2. The company's operating margin has been declining over the past few years, indicating potential challenges in maintaining profitability.\n3. The company's reliance on credit card transactions may expose it to fluctuations in the global economy and interest rates.\n\n[Prediction & Analysis]\nPrediction: AXP's stock price is likely to increase in the upcoming week, driven by the company's positive developments and the overall market trend.\n\nAnalysis: American Express has demonstrated a strong track record of innovation and growth, with a focus on enhancing customer experience and expanding its services. The company's recent dividend declaration and emphasis on generative AI suggest a commitment to rewarding shareholders and driving long-term growth. While there are potential concerns about the company's debt levels and operating margin, these issues are not likely to significantly impact the company's short-term performance. Overall, AXP's positive developments and strong financial performance make it a attractive investment opportunity, and its stock price is likely to continue its upward trend in the coming week.",
