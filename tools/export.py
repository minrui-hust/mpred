from mai.tools.export import main, parse_args

import mpred.data
import mpred.model

r'''
Evaluate model on evaluation set.
Optionally save predict output and evaluation metric
'''

if __name__ == '__main__':
    main(parse_args())
