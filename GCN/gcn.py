import fedml
import logging

from data.citation_networks import data_load


if __name__ == "__main__":
    logging.basicConfig(filename='gcn.log', encoding='utf-8', level=logging.INFO)

    args =fedml.init()
    logging.info('fedml init ~ ')
    

    device = fedml.device.get_device(args)
    
    dataset = data_load(args)

    # device = fedml.device.get_device(args)
