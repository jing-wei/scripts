# Not a function
# def pay_minimum(balance, annualInterestRate, monthlyPaymentRate):
balance = 4773; annualInterestRate = 0.2
#balance = 3283; annualInterestRate = 0.04
#balance = 3283; annualInterestRate = 0.04

lowest_pmt = int(10)
end_bal = balance
while end_bal >0: 
    end_bal = balance
    
    for month in range(1, 13): 
        end_bal = end_bal - lowest_pmt
        new_bal = end_bal + end_bal * annualInterestRate/12.0
        end_bal = new_bal
    if end_bal > 0:
        lowest_pmt += 10
    
    
print('Lowest payment: %s' % lowest_pmt)
