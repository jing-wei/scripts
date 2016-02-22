#
balance = 320000
annualInterestRate = 0.2
#balance = 999999
#annualInterestRate = 0.18

lowerBound = balance / 12.0
upperBound = (balance*(1+ annualInterestRate/12.0)**12) / 12.0
lowest_pmt = (lowerBound + upperBound) / 2
end_bal = balance
while (end_bal >0.02 or end_bal < -0.02): 
    end_bal = balance
    for month in range(1, 13): 
        end_bal = end_bal - lowest_pmt
        new_bal = end_bal + end_bal * (annualInterestRate/12.0)
        end_bal = new_bal
    if end_bal > 0.02:
        lowerBound = lowest_pmt
        lowest_pmt = (upperBound + lowerBound) / 2
    if end_bal < (-0.02):
        upperBound = lowest_pmt
        lowest_pmt = (upperBound + lowerBound) / 2    
print('Lowest payment: %s' % round(lowest_pmt, 2))