# counting and grouping
def item_order(order):
    total = dict(salad=0, hamburger=0, water=0)
    items = order.split()
    for item in items:
        if item == 'salad':
            total['salad'] += 1
        if item == 'hamburger':
            total['hamburger'] += 1
        if item == 'water':
            total['water'] += 1
    return 'salad:%(salad)s hamburger:%(hamburger)s water:%(water)s' % total