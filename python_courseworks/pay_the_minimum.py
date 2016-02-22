# 
def pay_minimum(balance, annualInterestRate, monthlyPaymentRate):
    paid = 0
    rem_bal = balance
    for month in range(1, 13):
        paid += monthlyPaymentRate*(balance)
        rem_bal = balance - monthlyPaymentRate*(balance)
        
        print('Month: %s' % month)
        print('Minimum monthly payment: %s' % \
        round(monthlyPaymentRate*(balance), 2))
        balance = rem_bal + rem_bal * annualInterestRate/12.0
        print('Remaining balance: %s' % \
        round(balance, 2))
        
    print('Total paid: %s' % round(paid,2))
    print('Remaining balance: %s' % round(balance,2))