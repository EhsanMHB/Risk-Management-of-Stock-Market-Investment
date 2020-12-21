# Risk Management of Stock Market Investment

This code has been developed as a part of a industrial project to automize risk management in stock market investment. It uses daily data of stock market including all stocks available in the market. After taking care of missing data, the code analyzes data to output:

  #### - Primary tables and plots:
At first, the code will provide different tables and plots through which the user can have a look at stock market behavior over time, prices, alpha, beta, risk, correlation between stocks etc. 

  #### - Optimized portfolios for the whole market: 
Using Markowitz model (please see: https://en.wikipedia.org/wiki/Markowitz_model), it proposes two optimized portfolios using all available and active stocks in the market; one based on the minimum risk and another one based on the maximum sharpe ratio. The optimized portfolios determine which stocks have to be selected and by what percentages.
  
  #### - Optimized portfolios for each individual sector: 
The same model has been employed to present two optimized portfolios for each individual sector, i.e. all available and active stocks in the industry. 

  #### - Predicted stock price: 
In addition to optimized portfolios, the code also applyes the Monte Carlo simulation (please see: https://en.wikipedia.org/wiki/Monte_Carlo_method) to predict the future prices for all stocks available in the market.


After analysis, the code updates the database by creating new tables to save the results properly.
