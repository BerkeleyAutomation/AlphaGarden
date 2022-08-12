import pickle
x = 10
radius = 0
priors = {
    "radicchio": [
        {
        'circle': ((1850, 350), 0, (1850, 350)),
        'days_post_germ': x,
        },
        {
            'circle': ((1950, 1300), 0, (1950, 1300)),
            'days_post_germ': x,
        }
    ],
    "turnip": [
        {
        'circle': ((2050, 715), 0, (2050, 715)),
        'days_post_germ': x,
        },
        {
            'circle': ((2550, 1020), 0, (2550, 1020)),
            'days_post_germ': x,
        }
    ],
    "cilantro": [
        {
        'circle': ((1850, 900), radius, (1850, 900)),
        'days_post_germ': x,
        },
        {
            'circle': ((2111, 982), radius, (2011, 920)),
            'days_post_germ': x,
        }
    ],
    "green-lettuce": [
        {
        'circle': ((1850, 1050), radius, (1850, 1050)),
        'days_post_germ': x,
        },
        {
            'circle': ((3000, 900), radius, (3000, 900)),
            'days_post_germ': x,
        }
    ],
    "red-lettuce": [
        {
        'circle': ((2200, 1400), radius, (2200, 1400)),
        'days_post_germ': x,
        },
        {
            'circle': ((3000, 700), radius, (3000, 700)),
            'days_post_germ': x,
        }
    ],
    "borage": [
        {
        'circle': ((2650, 400), radius, (2650, 400)),
        'days_post_germ': x,
        },
        {
            'circle': ((2500, 1200), radius, (2500, 1200)),
            'days_post_germ': x,
        }
    ],
    "kale": [
        {
        'circle': ((2870, 1200), radius, (2870, 1200)), #done
        'days_post_germ': x,
        },
        {
            'circle': ((2433, 688), radius, (2433, 688)), #done
            'days_post_germ': x,
        }
    ],
    "swiss-chard": [
        {
        'circle': ((2200, 490), radius, (2200, 490)),
        'days_post_germ': x,
        },
        {
            'circle': ((2800, 410), radius, (2800, 410)),
            'days_post_germ': x,
        }
    ],
}

f = open('out/priors/right/priorsinit.p', 'wb')   # Pickle file is newly created where foo1.py is
pickle.dump(priors, f)

# with open('out/priors/left/priorsinit.p', 'rb') as f:
#     print(pickle.load(f))
f = open('timestep.p', 'wb')   # Pickle file is newly created where foo1.py is
pickle.dump(30, f)
